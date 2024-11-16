use clap::ValueEnum;
use k8s_openapi::{
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        core::v1::ObjectReference,
    },
    apimachinery::pkg::apis::meta::v1::MicroTime,
    Resource,
};
use kube::{api::PostParams, Client, ResourceExt};
use resources::{inferenceservice::InferenceService, notebook::Notebook};
use secrecy::ExposeSecret;
use serde::Serialize;
use std::{hash::{Hash, Hasher}, path::Path};
use thiserror::Error;

use uuid::Uuid;

use std::fmt::Debug;

use prometheus_http_query::{response::InstantVector, Client as PromClient};
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;

use k8s_openapi::{
    api::core::v1::Event,
    apimachinery::pkg::apis::meta::v1::Time,
    chrono::{offset, Duration},
};
use kube::{
    api::{ObjectMeta, Patch, PatchParams},
    Api, Client as KubeClient,
};
use bitflags::bitflags;


#[derive(Debug, Clone, Serialize)]
pub enum ScaleKind {
    Deployment(Deployment),
    ReplicaSet(ReplicaSet),
    StatefulSet(StatefulSet),
    InferenceService(InferenceService),
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
    #[derive(Debug)]
    pub struct ResourceKind: u8 {
        const DEPLOYMENT = 0b00001;
        const REPLICA_SET = 0b00010;
        const STATEFUL_SET = 0b00100;
        const INFERENCE_SERVICE = 0b01000;
        const NOTEBOOK = 0b10000;
    }
}



pub struct QueryResposne {
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
    return Ok(std::str::from_utf8(&token)?.trim().to_string());
}


#[derive(Debug, Clone, Copy, ValueEnum, Default, Serialize)]
pub enum TlsMode {
    Skip,
    #[default]
    Verify,
}


pub fn get_prom_client<P: AsRef<Path>>(url: &str, token: String, verify_tls: TlsMode, certfile: Option<P>) -> anyhow::Result<PromClient> {
    let mut r_client = reqwest::ClientBuilder::new();

    if let TlsMode::Skip = verify_tls {
        r_client = r_client.danger_accept_invalid_certs(true);
    }

    if let Some(certfile) = certfile {
        r_client = r_client.add_root_certificate(reqwest::Certificate::from_pem(&std::fs::read(certfile)?)?);
    }

    // add auth token as default header
    let mut header_map = HeaderMap::new();
    header_map.insert("Authorization", format!("Bearer {}", token).parse()?);

    r_client = r_client.default_headers(header_map);

    let res = PromClient::from(r_client.build()?, url)?;

    Ok(res)
}

impl Meta for ScaleKind {
    fn name(&self) -> String {
        match self {
            ScaleKind::Deployment(d) => d.name_unchecked(),
            ScaleKind::ReplicaSet(d) => d.name_unchecked(),
            ScaleKind::StatefulSet(d) => d.name_unchecked(),
            ScaleKind::Notebook(d) => d.name_unchecked(),
            ScaleKind::InferenceService(d) => d.name_unchecked(),
        }
    }

    fn namespace(&self) -> Option<String> {
        match self {
            ScaleKind::Deployment(d) => d.namespace(),
            ScaleKind::ReplicaSet(d) => d.namespace(),
            ScaleKind::StatefulSet(d) => d.namespace(),
            ScaleKind::Notebook(d) => d.namespace(),
            ScaleKind::InferenceService(d) => d.namespace(),
        }
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
        match self {
            ScaleKind::Deployment(d) => d.uid(),
            ScaleKind::ReplicaSet(d) => d.uid(),
            ScaleKind::StatefulSet(d) => d.uid(),
            ScaleKind::Notebook(d) => d.uid(),
            ScaleKind::InferenceService(d) => d.uid(),
        }
    }

    fn resource_version(&self) -> Option<String> {
        match self {
            ScaleKind::Deployment(d) => d.resource_version(),
            ScaleKind::ReplicaSet(d) => d.resource_version(),
            ScaleKind::StatefulSet(d) => d.resource_version(),
            ScaleKind::Notebook(d) => d.resource_version(),
            ScaleKind::InferenceService(d) => d.resource_version(),
        }
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
                let _ = scale_notebook_to_zero(
                    client.clone(),
                    &d.name_unchecked(),
                    &d.namespace().expect("No namespace!"),
                )
                .await;
                Ok(())
            }
            ScaleKind::InferenceService(d) => {
                let _ = scale_inference_service_to_zero(
                    client.clone(),
                    &d.name_unchecked(),
                    &d.namespace().expect("No namespace!"),
                )
                .await;
                Ok(())
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn generate_scale_event(&self) -> anyhow::Result<Event> {
        let uuid = Uuid::new_v4();
        let now: Time = Time(offset::Utc::now());

        // this is intended to be set via pushdown
        let reporting_instance =
            Some(std::env::var("POD_NAME").unwrap_or("gpu_pruner".to_string()));

        let event: Event = Event {
            last_timestamp: Some(now.clone()),
            first_timestamp: Some(now.clone()),
            reporting_component: Some("gpu-pruner".to_string()),
            reporting_instance,
            event_time: Some(MicroTime(offset::Utc::now())),
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

/// Generic function to scale a resource to zero replicas
#[tracing::instrument(skip(api))]
async fn scale_to_zero<K>(api: Api<K>, name: &str) -> anyhow::Result<()>
where
    K: DeserializeOwned + Debug + Clone + kube::Resource<DynamicType = ()> + 'static,
{
    let patch = serde_json::json!({ "spec": { "replicas": 0 } });
    let params = PatchParams::default();
    api.patch(name, &params, &Patch::Strategic(&patch)).await?;
    Ok(())
}

/// Special handling for scaling a notebook to zero
///
/// TODO: maybe clean up the error handing or logging a bit, actually return the Resource.
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
                "kubeflow-resource-stopped": offset::Utc::now().to_rfc3339(),
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

    use kube::api::ObjectMeta;
    use resources::notebook::NotebookSpec;

    use super::{Notebook, ScaleKind};
    use crate::Scaler;

    #[test]
    fn make_event() {
        let sk: ScaleKind = ScaleKind::Notebook(Notebook {
            metadata: ObjectMeta {
                name: Some("gpu-test".to_string()),
                namespace: Some("rhoai-internal--weaton-nb".to_string()),
                ..Default::default()
            },
            spec: NotebookSpec { template: None },
            status: None,
        });

        let event = sk.generate_scale_event().expect("bar");

        println!("{}", serde_yaml::to_string(&event).expect("foo"));

        assert_eq!(event.involved_object.name, Some("gpu-test".to_string()))
    }
}
