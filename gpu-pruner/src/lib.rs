use clap::ValueEnum;
use k8s_openapi::{
    Resource,
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        core::v1::ObjectReference,
    },
    apimachinery::pkg::apis::meta::v1::MicroTime,
};
use kube::{Client, ResourceExt, api::PostParams};
use resources::{inferenceservice::InferenceService, notebook::Notebook};
use secrecy::ExposeSecret;
use serde::Serialize;
use std::{
    hash::{Hash, Hasher},
    path::Path,
};
use thiserror::Error;

use uuid::Uuid;

use std::fmt::Debug;

use prometheus_http_query::{Client as PromClient, response::InstantVector};
use reqwest::{Certificate, header::HeaderMap};
use serde::de::DeserializeOwned;

use bitflags::bitflags;
use jiff::Timestamp;
use k8s_openapi::{api::core::v1::Event, apimachinery::pkg::apis::meta::v1::Time};
use kube::{
    Api, Client as KubeClient,
    api::{ObjectMeta, Patch, PatchParams},
};

#[derive(Debug, Clone, Serialize)]
pub enum ScaleKind {
    Deployment(Deployment),
    ReplicaSet(ReplicaSet),
    StatefulSet(StatefulSet),
    InferenceService(Box<InferenceService>),
    Notebook(Notebook),
}

impl PartialEq for ScaleKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScaleKind::Deployment(a), ScaleKind::Deployment(b)) => a == b,
            (ScaleKind::ReplicaSet(a), ScaleKind::ReplicaSet(b)) => a == b,
            (ScaleKind::StatefulSet(a), ScaleKind::StatefulSet(b)) => a == b,
            (ScaleKind::InferenceService(a), ScaleKind::InferenceService(b)) => a.uid() == b.uid(),
            (ScaleKind::Notebook(a), ScaleKind::Notebook(b)) => a.uid() == b.uid(),
            // If they are different variants, they are not equal
            _ => false,
        }
    }
}
impl Eq for ScaleKind {}

impl Hash for ScaleKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Differentiate between different enum variants
        std::mem::discriminant(self).hash(state);
        match self {
            ScaleKind::Deployment(a) => {
                a.uid().hash(state);
            }
            ScaleKind::ReplicaSet(a) => {
                a.uid().hash(state);
            }
            ScaleKind::StatefulSet(a) => {
                a.uid().hash(state);
            }
            ScaleKind::InferenceService(a) => {
                a.uid().hash(state);
            }
            ScaleKind::Notebook(a) => {
                a.uid().hash(state);
            }
        }
    }
}

impl From<ScaleKind> for ResourceKind {
    fn from(kind: ScaleKind) -> Self {
        match kind {
            ScaleKind::Deployment(_) => ResourceKind::DEPLOYMENT,
            ScaleKind::ReplicaSet(_) => ResourceKind::REPLICA_SET,
            ScaleKind::StatefulSet(_) => ResourceKind::STATEFUL_SET,
            ScaleKind::InferenceService(_) => ResourceKind::INFERENCE_SERVICE,
            ScaleKind::Notebook(_) => ResourceKind::NOTEBOOK,
        }
    }
}

bitflags! {
    #[derive(Debug, PartialEq, Eq)]
    pub struct ResourceKind: u8 {
        const DEPLOYMENT = 0b00001;
        const REPLICA_SET = 0b00010;
        const STATEFUL_SET = 0b00100;
        const INFERENCE_SERVICE = 0b01000;
        const NOTEBOOK = 0b10000;
    }
}

/// Parse a string of resource flag characters into a [`ResourceKind`] bitflag set.
///
/// - `d` → Deployment
/// - `r` → ReplicaSet
/// - `s` → StatefulSet
/// - `i` → InferenceService
/// - `n` → Notebook
///
/// Unknown characters are silently ignored.
pub fn get_enabled_resources(enabled_resources: &str) -> ResourceKind {
    let mut resource_kind = ResourceKind::empty();
    for c in enabled_resources.chars() {
        match c {
            'd' => resource_kind |= ResourceKind::DEPLOYMENT,
            'r' => resource_kind |= ResourceKind::REPLICA_SET,
            's' => resource_kind |= ResourceKind::STATEFUL_SET,
            'i' => resource_kind |= ResourceKind::INFERENCE_SERVICE,
            'n' => resource_kind |= ResourceKind::NOTEBOOK,
            _ => {}
        }
    }
    resource_kind
}

pub struct QueryResponse {
    pub num_pods: usize,
    pub shutdown_events: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct PodMetricData {
    pub name: String,
    pub namespace: String,
    pub container: String,
    pub node_type: String,
    pub gpu_model: String,
    pub value: f64,
}

#[derive(Debug, Error)]
pub enum PodConvertError {
    #[error("the data for key `{0}` is not available")]
    UnwrapError(String),
}

impl TryFrom<&InstantVector> for PodMetricData {
    type Error = PodConvertError;

    fn try_from(value: &InstantVector) -> Result<Self, PodConvertError> {
        let metrics = value.metric();
        tracing::debug!("Metrics: {metrics:#?}");

        Ok(PodMetricData {
            name: metrics
                .get("exported_pod")
                .ok_or_else(|| PodConvertError::UnwrapError("exported_pod".into()))?
                .clone(),
            namespace: metrics
                .get("exported_namespace")
                .ok_or_else(|| PodConvertError::UnwrapError("exported_namespace".into()))?
                .clone(),
            container: metrics
                .get("exported_container")
                .ok_or_else(|| PodConvertError::UnwrapError("exported_container".into()))?
                .clone(),
            node_type: metrics
                .get("node_type")
                .ok_or_else(|| PodConvertError::UnwrapError("node_type".into()))?
                .clone(),
            gpu_model: metrics
                .get("modelName")
                .ok_or_else(|| PodConvertError::UnwrapError("modelName".into()))?
                .clone(),
            value: value.sample().value(),
        })
    }
}
pub trait Meta {
    fn name(&self) -> String;
    fn namespace(&self) -> Option<String>;
    fn kind(&self) -> String;
    fn uid(&self) -> Option<String>;
    fn api_version(&self) -> String;
    fn resource_version(&self) -> Option<String>;
}

pub trait Scaler {
    fn scale(&self, client: Client)
    -> impl std::future::Future<Output = anyhow::Result<()>> + Send;

    fn generate_scale_event(&self) -> anyhow::Result<Event>;
}

/// Get the token for the prometheus client
pub async fn get_prometheus_token() -> anyhow::Result<String> {
    if let Ok(token) = std::env::var("PROMETHEUS_TOKEN") {
        tracing::debug!("Getting token from PROMETHEUS_TOKEN");
        return Ok(token);
    }

    tracing::info!("Inferring prometheus token from K8s config");
    let config: kube::Config = kube::Config::infer().await?;
    tracing::trace!("{config:#?}");
    // in cluster config usually causes a token file
    if let Some(token_file) = config.auth_info.token_file {
        let token = std::fs::read_to_string(token_file)?;
        return Ok(token);
    }
    if let Some(token) = config.auth_info.token {
        tracing::info!("Found K8s token");
        let token = token.expose_secret();
        return Ok(token.to_string());
    }

    tracing::info!("No token provided, trying to get token from oc as last resort");
    let token = std::process::Command::new("oc")
        .args(["whoami", "-t"])
        .output()?
        .stdout;
    Ok(std::str::from_utf8(&token)?.trim().to_string())
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, Serialize)]
pub enum TlsMode {
    Skip,
    #[default]
    Verify,
}

pub fn get_prom_client<P: AsRef<Path>>(
    url: &str,
    token: String,
    verify_tls: TlsMode,
    certfile: Option<P>,
) -> anyhow::Result<PromClient> {
    let mut r_client = reqwest::ClientBuilder::new();

    if let TlsMode::Skip = verify_tls {
        r_client = r_client.danger_accept_invalid_certs(true);
    }

    if let Some(certfile) = certfile {
        if let Ok(cert_data) = std::fs::read(certfile) {
            tracing::debug!("Read certificate file");
            if let Ok(certs) = Certificate::from_pem_bundle(&cert_data) {
                tracing::debug!("Parsed certificates from PEM bundle");
                for cert in certs {
                    tracing::debug!("Adding root certificate");
                    r_client = r_client.add_root_certificate(cert);
                }
            } else {
                tracing::error!("Failed to parse certificates from PEM bundle");
                return Err(anyhow::anyhow!(
                    "Failed to parse certificates from PEM bundle"
                ));
            }
        } else {
            tracing::error!("Failed to read certificate file");
            return Err(anyhow::anyhow!("Failed to read certificate file"));
        }
    }

    // add auth token as default header
    let mut header_map = HeaderMap::new();
    header_map.insert("Authorization", format!("Bearer {}", token).parse()?);

    r_client = r_client.default_headers(header_map);

    let res = PromClient::from(r_client.build()?, url)?;

    Ok(res)
}

/// Dispatches a method call through every ScaleKind variant.
/// All inner types implement ResourceExt, so this avoids repeating
/// the same match block for each delegating Meta method.
macro_rules! delegate_resource_ext {
    ($self:expr, $method:ident) => {
        match $self {
            ScaleKind::Deployment(d) => d.$method(),
            ScaleKind::ReplicaSet(d) => d.$method(),
            ScaleKind::StatefulSet(d) => d.$method(),
            ScaleKind::InferenceService(d) => d.$method(),
            ScaleKind::Notebook(d) => d.$method(),
        }
    };
}

impl Meta for ScaleKind {
    fn name(&self) -> String {
        delegate_resource_ext!(self, name_unchecked)
    }

    fn namespace(&self) -> Option<String> {
        delegate_resource_ext!(self, namespace)
    }

    fn api_version(&self) -> String {
        match self {
            ScaleKind::Deployment(_) => Deployment::API_VERSION.to_string(),
            ScaleKind::ReplicaSet(_) => ReplicaSet::API_VERSION.to_string(),
            ScaleKind::StatefulSet(_) => StatefulSet::API_VERSION.to_string(),
            ScaleKind::Notebook(_) => "v1".to_string(),
            ScaleKind::InferenceService(_) => "v1beta1".to_string(),
        }
    }

    fn kind(&self) -> String {
        match self {
            ScaleKind::Deployment(_) => Deployment::KIND.to_string(),
            ScaleKind::ReplicaSet(_) => ReplicaSet::KIND.to_string(),
            ScaleKind::StatefulSet(_) => StatefulSet::KIND.to_string(),
            ScaleKind::Notebook(_) => "Notebook".to_string(),
            ScaleKind::InferenceService(_) => "InferenceService".to_string(),
        }
    }

    fn uid(&self) -> Option<String> {
        delegate_resource_ext!(self, uid)
    }

    fn resource_version(&self) -> Option<String> {
        delegate_resource_ext!(self, resource_version)
    }
}

impl Scaler for ScaleKind {
    #[tracing::instrument(skip(self, client))]
    async fn scale(&self, client: Client) -> anyhow::Result<()> {
        if let Some(ns) = self.namespace() {
            let event = self.generate_scale_event()?;
            let events_api: Api<Event> = Api::namespaced(client.clone(), &ns);

            if let Err(e) = events_api.create(&PostParams::default(), &event).await {
                tracing::error!("Failed to push Event for scale down!: {e}");
            } else {
                tracing::debug!("Emitted scale event for: {:?}", event.involved_object);
            }
        };

        match self {
            ScaleKind::Deployment(d) => {
                let api: Api<Deployment> =
                    Api::namespaced(client.clone(), &d.namespace().expect("No namespace!"));
                scale_to_zero(api, &d.name_unchecked()).await
            }
            ScaleKind::ReplicaSet(d) => {
                let api: Api<ReplicaSet> =
                    Api::namespaced(client.clone(), &d.namespace().expect("No namespace!"));
                scale_to_zero(api, &d.name_unchecked()).await
            }
            ScaleKind::StatefulSet(d) => {
                let api: Api<StatefulSet> =
                    Api::namespaced(client.clone(), &d.namespace().expect("No namespace!"));
                scale_to_zero(api, &d.name_unchecked()).await
            }
            ScaleKind::Notebook(d) => {
                scale_notebook_to_zero(
                    client.clone(),
                    &d.name_unchecked(),
                    &d.namespace().expect("No namespace!"),
                )
                .await?;
                Ok(())
            }
            ScaleKind::InferenceService(d) => {
                scale_inference_service_to_zero(
                    client.clone(),
                    &d.name_unchecked(),
                    &d.namespace().expect("No namespace!"),
                )
                .await?;
                Ok(())
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn generate_scale_event(&self) -> anyhow::Result<Event> {
        let uuid = Uuid::new_v4();
        let now: Time = Time(Timestamp::now());

        // this is intended to be set via pushdown
        let reporting_instance =
            Some(std::env::var("POD_NAME").unwrap_or("gpu_pruner".to_string()));

        let event: Event = Event {
            last_timestamp: Some(now.clone()),
            first_timestamp: Some(now.clone()),
            reporting_component: Some("gpu-pruner".to_string()),
            reporting_instance,
            event_time: Some(MicroTime(Timestamp::now())),
            action: Some("scale_down".to_string()),
            reason: Some(format!(
                "Pod {}::{} was not using GPU",
                self.namespace().unwrap_or("".to_string()),
                self.name()
            )),
            type_: Some("Normal".to_string()),
            metadata: ObjectMeta {
                namespace: self.namespace(),
                name: Some(format!("gpuscaler-{}", uuid.as_simple())),
                ..Default::default()
            },
            involved_object: ObjectReference {
                api_version: Some(self.api_version()),
                field_path: None,
                kind: Some(self.kind()),
                name: Some(self.name()),
                namespace: self.namespace(),
                resource_version: self.resource_version(),
                uid: self.uid(),
            },
            ..Default::default()
        };
        Ok(event)
    }
}

/// Crawl up the owner references to find the root Deployment or StatefulSet
/// and allows an action like scaling to be performed.
///
/// Deployments and StatefulSets can have multiple pods, so we shouldn't
/// "double scale-down" them if they share a common parent and both pods
/// do not have GPU utilization. We only need to send the request once.
#[tracing::instrument(skip(client, pod_meta), fields(name = pod_meta.name))]
pub async fn find_root_object(
    client: KubeClient,
    pod_meta: &ObjectMeta,
) -> anyhow::Result<ScaleKind> {
    tracing::info!(
        "Finding root object of {name:?} for scale-down.",
        name = &pod_meta.name
    );
    // first, check for the special kserve label
    // if it exists, we can go directly to the InferenceService
    // and scale it down
    if let Some(labels) = &pod_meta.labels
        && let Some(ks_label) = labels.get("serving.kserve.io/inferenceservice")
    {
        let namespace = pod_meta.namespace.clone().unwrap_or_default();
        let is_api: Api<InferenceService> = Api::namespaced(client.clone(), &namespace);
        let is = is_api.get(ks_label).await?;

        return Ok(ScaleKind::InferenceService(Box::new(is)));
    }

    if let Some(ors) = &pod_meta.owner_references {
        for or in ors {
            let namespace = pod_meta.namespace.clone().unwrap_or_default();
            match or.kind.as_str() {
                "ReplicaSet" => {
                    tracing::info!("Found ReplicaSet!");
                    let rs_api: Api<ReplicaSet> = Api::namespaced(client.clone(), &namespace);
                    if let Ok(rs) = rs_api.get(&or.name).await {
                        if let Some(rs_meta) = rs.metadata.owner_references.as_ref() {
                            for rs_or in rs_meta {
                                if rs_or.kind == "Deployment" {
                                    tracing::info!("Found Deployment owning ReplicaSet!");
                                    let deployment_api: Api<Deployment> =
                                        Api::namespaced(client.clone(), &namespace);
                                    let deployment = deployment_api.get(&rs_or.name).await?;

                                    return Ok(ScaleKind::Deployment(deployment));
                                }
                            }
                        }
                        // fallthrough, replica set with no owners
                        return Ok(ScaleKind::ReplicaSet(rs.clone()));
                    }
                }
                "StatefulSet" => {
                    tracing::info!("Found StatefulSet!");
                    let ss_api: Api<StatefulSet> = Api::namespaced(client.clone(), &namespace);
                    if let Ok(ss) = ss_api.get(&or.name).await {
                        if let Some(ss_meta) = ss.metadata.owner_references.as_ref() {
                            for ss_or in ss_meta {
                                if ss_or.kind == "Notebook" {
                                    tracing::info!("Found Notebook owning StatefulSet!");
                                    let nb_api: Api<Notebook> =
                                        Api::namespaced(client.clone(), &namespace);
                                    let nb = nb_api.get(&ss_or.name).await?;

                                    return Ok(ScaleKind::Notebook(nb));
                                }
                            }
                        }
                        // fallthrough, statefulset with no owners
                        return Ok(ScaleKind::StatefulSet(ss));
                    }
                }
                _ => {
                    tracing::warn!("Found no ORs!")
                }
            }
        }
    }

    Err(anyhow::anyhow!("oops, nothing found!"))
}

/// Scale a resource to zero replicas via the /scale subresource endpoint
#[tracing::instrument(skip(api))]
async fn scale_to_zero<K>(api: Api<K>, name: &str) -> anyhow::Result<()>
where
    K: DeserializeOwned + Debug + Clone + kube::Resource<DynamicType = ()> + 'static,
{
    let patch = serde_json::json!({ "spec": { "replicas": 0 } });
    api.patch_scale(name, &PatchParams::default(), &Patch::Merge(&patch))
        .await?;
    Ok(())
}

/// Special handling for scaling a notebook to zero
#[tracing::instrument(skip(client))]
async fn scale_notebook_to_zero(
    client: KubeClient,
    name: &str,
    namespace: &str,
) -> anyhow::Result<Notebook> {
    let notebook_api: Api<Notebook> = Api::namespaced(client.clone(), namespace);

    let patch = serde_json::json!({
        "metadata": {
            "annotations": {
                "kubeflow-resource-stopped": Timestamp::now().to_string(),
            }
        }
    });

    let res = notebook_api
        .patch(name, &PatchParams::default(), &Patch::Merge(patch))
        .await?;

    Ok(res)
}

/// Special handling for scaling an InferenceService to zero
#[tracing::instrument(skip(client))]
async fn scale_inference_service_to_zero(
    client: KubeClient,
    name: &str,
    namespace: &str,
) -> anyhow::Result<InferenceService> {
    let is_api: Api<InferenceService> = Api::namespaced(client.clone(), namespace);

    // by setting spec.predictor.minReplicas to 0 we allow KServe to scale down the request for us, it will get
    // rescaled automatically when traffic kicks back up. the user will need to manually set minReplicas
    // back to 1 though to get durable capacity again.
    let patch = serde_json::json!({
        "spec": {
            "predictor": {
                "minReplicas": 0
            }
        }
    });

    let res = is_api
        .patch(name, &PatchParams::default(), &Patch::Merge(patch))
        .await?;

    Ok(res)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use k8s_openapi::api::apps::v1::{Deployment, ReplicaSet, StatefulSet};
    use kube::api::ObjectMeta;
    use resources::{inferenceservice::InferenceService, notebook::NotebookSpec};

    use crate::{Meta, Notebook, ResourceKind, ScaleKind, Scaler, get_enabled_resources};

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_notebook(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        ScaleKind::Notebook(Notebook {
            metadata: ObjectMeta {
                name: Some(name.into()),
                namespace: Some(ns.into()),
                uid: uid.map(Into::into),
                ..Default::default()
            },
            spec: NotebookSpec { template: None },
            status: None,
        })
    }

    fn make_deployment(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        ScaleKind::Deployment(Deployment {
            metadata: ObjectMeta {
                name: Some(name.into()),
                namespace: Some(ns.into()),
                uid: uid.map(Into::into),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn make_replica_set(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        ScaleKind::ReplicaSet(ReplicaSet {
            metadata: ObjectMeta {
                name: Some(name.into()),
                namespace: Some(ns.into()),
                uid: uid.map(Into::into),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn make_stateful_set(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        ScaleKind::StatefulSet(StatefulSet {
            metadata: ObjectMeta {
                name: Some(name.into()),
                namespace: Some(ns.into()),
                uid: uid.map(Into::into),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    fn make_inference_service(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        let mut is: InferenceService = serde_json::from_value(serde_json::json!({
            "metadata": {
                "name": name,
                "namespace": ns,
            },
            "spec": {
                "predictor": {}
            }
        }))
        .expect("valid InferenceService JSON");
        is.metadata.uid = uid.map(Into::into);
        ScaleKind::InferenceService(Box::new(is))
    }

    // ── get_enabled_resources ────────────────────────────────────────────

    #[test]
    fn enabled_resources_all_flags() {
        let rk = get_enabled_resources("drsin");
        assert!(rk.contains(ResourceKind::DEPLOYMENT));
        assert!(rk.contains(ResourceKind::REPLICA_SET));
        assert!(rk.contains(ResourceKind::STATEFUL_SET));
        assert!(rk.contains(ResourceKind::INFERENCE_SERVICE));
        assert!(rk.contains(ResourceKind::NOTEBOOK));
    }

    #[test]
    fn enabled_resources_single_flag() {
        let rk = get_enabled_resources("n");
        assert!(rk.contains(ResourceKind::NOTEBOOK));
        assert!(!rk.contains(ResourceKind::DEPLOYMENT));
        assert!(!rk.contains(ResourceKind::REPLICA_SET));
        assert!(!rk.contains(ResourceKind::STATEFUL_SET));
        assert!(!rk.contains(ResourceKind::INFERENCE_SERVICE));
    }

    #[test]
    fn enabled_resources_subset() {
        let rk = get_enabled_resources("di");
        assert!(rk.contains(ResourceKind::DEPLOYMENT));
        assert!(rk.contains(ResourceKind::INFERENCE_SERVICE));
        assert!(!rk.contains(ResourceKind::NOTEBOOK));
        assert!(!rk.contains(ResourceKind::REPLICA_SET));
        assert!(!rk.contains(ResourceKind::STATEFUL_SET));
    }

    #[test]
    fn enabled_resources_empty_string() {
        let rk = get_enabled_resources("");
        assert!(rk.is_empty());
    }

    #[test]
    fn enabled_resources_ignores_unknown_chars() {
        let rk = get_enabled_resources("xdqz");
        assert!(rk.contains(ResourceKind::DEPLOYMENT));
        assert!(!rk.contains(ResourceKind::NOTEBOOK));
    }

    #[test]
    fn enabled_resources_duplicate_chars_are_idempotent() {
        let rk = get_enabled_resources("dddd");
        assert_eq!(rk, get_enabled_resources("d"));
    }

    // ── ResourceKind bitflags ────────────────────────────────────────────

    #[test]
    fn resource_kind_union() {
        let combined = ResourceKind::DEPLOYMENT | ResourceKind::NOTEBOOK;
        assert!(combined.contains(ResourceKind::DEPLOYMENT));
        assert!(combined.contains(ResourceKind::NOTEBOOK));
        assert!(!combined.contains(ResourceKind::STATEFUL_SET));
    }

    #[test]
    fn resource_kind_empty_contains_nothing() {
        let empty = ResourceKind::empty();
        assert!(!empty.contains(ResourceKind::DEPLOYMENT));
        assert!(!empty.contains(ResourceKind::REPLICA_SET));
        assert!(!empty.contains(ResourceKind::STATEFUL_SET));
        assert!(!empty.contains(ResourceKind::INFERENCE_SERVICE));
        assert!(!empty.contains(ResourceKind::NOTEBOOK));
    }

    // ── ScaleKind → ResourceKind conversion ──────────────────────────────

    #[test]
    fn scale_kind_to_resource_kind_deployment() {
        let rk: ResourceKind = make_deployment("d", "ns", None).into();
        assert_eq!(rk, ResourceKind::DEPLOYMENT);
    }

    #[test]
    fn scale_kind_to_resource_kind_replica_set() {
        let rk: ResourceKind = make_replica_set("r", "ns", None).into();
        assert_eq!(rk, ResourceKind::REPLICA_SET);
    }

    #[test]
    fn scale_kind_to_resource_kind_stateful_set() {
        let rk: ResourceKind = make_stateful_set("s", "ns", None).into();
        assert_eq!(rk, ResourceKind::STATEFUL_SET);
    }

    #[test]
    fn scale_kind_to_resource_kind_inference_service() {
        let rk: ResourceKind = make_inference_service("i", "ns", None).into();
        assert_eq!(rk, ResourceKind::INFERENCE_SERVICE);
    }

    #[test]
    fn scale_kind_to_resource_kind_notebook() {
        let rk: ResourceKind = make_notebook("n", "ns", None).into();
        assert_eq!(rk, ResourceKind::NOTEBOOK);
    }

    // ── ScaleKind equality ───────────────────────────────────────────────

    #[test]
    fn same_deployment_is_equal() {
        let a = make_deployment("d", "ns", Some("uid-1"));
        let b = make_deployment("d", "ns", Some("uid-1"));
        assert_eq!(a, b);
    }

    #[test]
    fn different_uid_deployments_not_equal() {
        let a = make_deployment("d", "ns", Some("uid-1"));
        let b = make_deployment("d", "ns", Some("uid-2"));
        assert_ne!(a, b);
    }

    #[test]
    fn different_variants_not_equal() {
        let dep = make_deployment("x", "ns", Some("uid-1"));
        let rs = make_replica_set("x", "ns", Some("uid-1"));
        assert_ne!(dep, rs);
    }

    #[test]
    fn notebook_equality_uses_uid() {
        let a = make_notebook("nb-a", "ns", Some("same-uid"));
        let b = make_notebook("nb-b", "ns", Some("same-uid"));
        assert_eq!(a, b);
    }

    #[test]
    fn inference_service_equality_uses_uid() {
        let a = make_inference_service("is-a", "ns", Some("uid-x"));
        let b = make_inference_service("is-b", "ns", Some("uid-x"));
        assert_eq!(a, b);
    }

    // ── ScaleKind hashing / HashSet dedup ────────────────────────────────

    #[test]
    fn hashset_deduplicates_same_deployment() {
        let mut set = HashSet::new();
        set.insert(make_deployment("d", "ns", Some("uid-1")));
        set.insert(make_deployment("d", "ns", Some("uid-1")));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn hashset_keeps_different_uid_deployments() {
        let mut set = HashSet::new();
        set.insert(make_deployment("d", "ns", Some("uid-1")));
        set.insert(make_deployment("d", "ns", Some("uid-2")));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn hashset_keeps_different_variants_same_uid() {
        let mut set = HashSet::new();
        set.insert(make_deployment("x", "ns", Some("uid-1")));
        set.insert(make_replica_set("x", "ns", Some("uid-1")));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn hashset_deduplicates_notebooks_by_uid() {
        let mut set = HashSet::new();
        set.insert(make_notebook("nb-a", "ns", Some("uid-nb")));
        set.insert(make_notebook("nb-b", "ns", Some("uid-nb")));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn hashset_mixed_resources() {
        let mut set = HashSet::new();
        set.insert(make_deployment("d1", "ns", Some("uid-d")));
        set.insert(make_replica_set("r1", "ns", Some("uid-r")));
        set.insert(make_stateful_set("s1", "ns", Some("uid-s")));
        set.insert(make_inference_service("i1", "ns", Some("uid-i")));
        set.insert(make_notebook("n1", "ns", Some("uid-n")));
        // duplicate of first deployment
        set.insert(make_deployment("d1", "ns", Some("uid-d")));
        assert_eq!(set.len(), 5);
    }

    // ── Meta trait ───────────────────────────────────────────────────────

    #[test]
    fn meta_deployment() {
        let sk = make_deployment("my-dep", "prod", Some("dep-uid"));
        assert_eq!(sk.name(), "my-dep");
        assert_eq!(sk.namespace(), Some("prod".into()));
        assert_eq!(sk.kind(), "Deployment");
        assert_eq!(sk.uid(), Some("dep-uid".into()));
        assert_eq!(sk.api_version(), "apps/v1");
    }

    #[test]
    fn meta_replica_set() {
        let sk = make_replica_set("my-rs", "staging", Some("rs-uid"));
        assert_eq!(sk.name(), "my-rs");
        assert_eq!(sk.namespace(), Some("staging".into()));
        assert_eq!(sk.kind(), "ReplicaSet");
        assert_eq!(sk.uid(), Some("rs-uid".into()));
        assert_eq!(sk.api_version(), "apps/v1");
    }

    #[test]
    fn meta_stateful_set() {
        let sk = make_stateful_set("my-ss", "dev", Some("ss-uid"));
        assert_eq!(sk.name(), "my-ss");
        assert_eq!(sk.namespace(), Some("dev".into()));
        assert_eq!(sk.kind(), "StatefulSet");
        assert_eq!(sk.uid(), Some("ss-uid".into()));
        assert_eq!(sk.api_version(), "apps/v1");
    }

    #[test]
    fn meta_notebook() {
        let sk = make_notebook("my-nb", "ml", Some("nb-uid"));
        assert_eq!(sk.name(), "my-nb");
        assert_eq!(sk.namespace(), Some("ml".into()));
        assert_eq!(sk.kind(), "Notebook");
        assert_eq!(sk.uid(), Some("nb-uid".into()));
        assert_eq!(sk.api_version(), "v1");
    }

    #[test]
    fn meta_inference_service() {
        let sk = make_inference_service("my-is", "serving", Some("is-uid"));
        assert_eq!(sk.name(), "my-is");
        assert_eq!(sk.namespace(), Some("serving".into()));
        assert_eq!(sk.kind(), "InferenceService");
        assert_eq!(sk.uid(), Some("is-uid".into()));
        assert_eq!(sk.api_version(), "v1beta1");
    }

    // ── Event generation ─────────────────────────────────────────────────

    #[test]
    fn event_for_notebook() {
        let sk = make_notebook("gpu-test", "rhoai--weaton", Some("nb-uid-1"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.name, Some("gpu-test".into()));
        assert_eq!(
            event.involved_object.namespace,
            Some("rhoai--weaton".into())
        );
        assert_eq!(event.involved_object.kind, Some("Notebook".into()));
        assert_eq!(event.involved_object.uid, Some("nb-uid-1".into()));
        assert_eq!(event.involved_object.api_version, Some("v1".into()));
        assert_eq!(event.action, Some("scale_down".into()));
        assert_eq!(event.type_, Some("Normal".into()));
        assert_eq!(
            event.reason,
            Some("Pod rhoai--weaton::gpu-test was not using GPU".into())
        );
        assert_eq!(event.reporting_component, Some("gpu-pruner".into()));
        assert!(event.metadata.name.unwrap().starts_with("gpuscaler-"));
        assert_eq!(event.metadata.namespace, Some("rhoai--weaton".into()));
        assert!(event.first_timestamp.is_some());
        assert!(event.last_timestamp.is_some());
        assert!(event.event_time.is_some());
    }

    #[test]
    fn event_for_deployment() {
        let sk = make_deployment("my-dep", "prod", Some("dep-uid"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.kind, Some("Deployment".into()));
        assert_eq!(event.involved_object.api_version, Some("apps/v1".into()));
        assert_eq!(
            event.reason,
            Some("Pod prod::my-dep was not using GPU".into())
        );
    }

    #[test]
    fn event_for_replica_set() {
        let sk = make_replica_set("my-rs", "staging", None);
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.kind, Some("ReplicaSet".into()));
        assert_eq!(event.involved_object.uid, None);
    }

    #[test]
    fn event_for_stateful_set() {
        let sk = make_stateful_set("my-ss", "dev", Some("ss-uid"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.kind, Some("StatefulSet".into()));
        assert_eq!(event.involved_object.api_version, Some("apps/v1".into()));
    }

    #[test]
    fn event_for_inference_service() {
        let sk = make_inference_service("my-is", "serving", Some("is-uid"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.kind, Some("InferenceService".into()));
        assert_eq!(event.involved_object.api_version, Some("v1beta1".into()));
    }

    #[test]
    fn event_names_are_unique() {
        let sk = make_notebook("nb", "ns", None);
        let e1 = sk.generate_scale_event().unwrap();
        let e2 = sk.generate_scale_event().unwrap();
        assert_ne!(e1.metadata.name, e2.metadata.name);
    }

    #[test]
    fn event_with_no_namespace() {
        let sk = ScaleKind::Deployment(Deployment {
            metadata: ObjectMeta {
                name: Some("orphan".into()),
                namespace: None,
                ..Default::default()
            },
            ..Default::default()
        });
        let event = sk.generate_scale_event().unwrap();
        assert_eq!(event.involved_object.namespace, None);
        assert_eq!(event.reason, Some("Pod ::orphan was not using GPU".into()));
    }

    // ── resource filtering integration ───────────────────────────────────

    #[test]
    fn enabled_resources_filter_accepts_matching_scale_kind() {
        let enabled = get_enabled_resources("dn");
        let dep: ResourceKind = make_deployment("d", "ns", None).into();
        let nb: ResourceKind = make_notebook("n", "ns", None).into();
        let ss: ResourceKind = make_stateful_set("s", "ns", None).into();

        assert!(enabled.contains(dep));
        assert!(enabled.contains(nb));
        assert!(!enabled.contains(ss));
    }
}
