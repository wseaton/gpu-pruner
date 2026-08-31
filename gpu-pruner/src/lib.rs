pub mod api;
pub mod metrics;
pub mod slack;

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
use resources::{
    inferenceservice::InferenceService, leaderworkerset::LeaderWorkerSet,
    llminferenceservice::LLMInferenceService, notebook::Notebook,
};
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

pub const PENDING_SCALE_ANNOTATION: &str = "gpu-pruner.io/pending-scale-at";
pub const ACK_UNTIL_ANNOTATION: &str = "gpu-pruner.io/ack-until";
pub const ACK_BY_ANNOTATION: &str = "gpu-pruner.io/ack-by";
pub const SLACK_MENTIONS_ANNOTATION: &str = "gpu-pruner.io/slack-mentions";

#[derive(Debug, Clone)]
pub struct AckStatus {
    pub acknowledged: bool,
    pub expires_at: Option<String>,
    pub by_user: Option<String>,
}

/// Maps namespaces to Slack mention strings. Keys ending in `-` are treated
/// as namespace prefixes; the longest matching prefix wins.
#[derive(Debug, Clone)]
pub struct NamespaceMentionMapper {
    pub mappings: std::collections::HashMap<String, String>,
}

impl NamespaceMentionMapper {
    pub fn new(mappings: std::collections::HashMap<String, String>) -> Self {
        Self { mappings }
    }

    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        let mappings: std::collections::HashMap<String, String> = serde_json::from_str(json_str)?;
        Ok(Self::new(mappings))
    }

    pub fn get_mentions(&self, namespace: &str) -> Option<String> {
        if let Some(mentions) = self.mappings.get(namespace) {
            return Some(mentions.clone());
        }

        self.mappings
            .iter()
            .filter(|(key, _)| key.ends_with('-') && namespace.starts_with(key.as_str()))
            .max_by_key(|(key, _)| key.len())
            .map(|(_, mentions)| mentions.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingScaleStatus {
    NotPending,
    InGrace {
        until: chrono::DateTime<chrono::Utc>,
    },
    GraceExpired,
    /// The pending annotation is so old the notification predates the current
    /// idle episode; treat like NotPending and re-notify.
    Stale,
}

#[derive(Debug, Clone, Serialize)]
pub enum ScaleKind {
    Deployment(Deployment),
    ReplicaSet(ReplicaSet),
    StatefulSet(StatefulSet),
    InferenceService(Box<InferenceService>),
    LLMInferenceService(Box<LLMInferenceService>),
    Notebook(Notebook),
    LeaderWorkerSet(LeaderWorkerSet),
}

impl PartialEq for ScaleKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ScaleKind::Deployment(a), ScaleKind::Deployment(b)) => a == b,
            (ScaleKind::ReplicaSet(a), ScaleKind::ReplicaSet(b)) => a == b,
            (ScaleKind::StatefulSet(a), ScaleKind::StatefulSet(b)) => a == b,
            (ScaleKind::InferenceService(a), ScaleKind::InferenceService(b)) => a.uid() == b.uid(),
            (ScaleKind::LLMInferenceService(a), ScaleKind::LLMInferenceService(b)) => {
                a.uid() == b.uid()
            }
            (ScaleKind::Notebook(a), ScaleKind::Notebook(b)) => a.uid() == b.uid(),
            (ScaleKind::LeaderWorkerSet(a), ScaleKind::LeaderWorkerSet(b)) => a.uid() == b.uid(),
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
            ScaleKind::LLMInferenceService(a) => {
                a.uid().hash(state);
            }
            ScaleKind::Notebook(a) => {
                a.uid().hash(state);
            }
            ScaleKind::LeaderWorkerSet(a) => {
                a.uid().hash(state);
            }
        }
    }
}

impl From<Deployment> for ScaleKind {
    fn from(v: Deployment) -> Self {
        ScaleKind::Deployment(v)
    }
}
impl From<ReplicaSet> for ScaleKind {
    fn from(v: ReplicaSet) -> Self {
        ScaleKind::ReplicaSet(v)
    }
}
impl From<StatefulSet> for ScaleKind {
    fn from(v: StatefulSet) -> Self {
        ScaleKind::StatefulSet(v)
    }
}
impl From<Notebook> for ScaleKind {
    fn from(v: Notebook) -> Self {
        ScaleKind::Notebook(v)
    }
}
impl From<LeaderWorkerSet> for ScaleKind {
    fn from(v: LeaderWorkerSet) -> Self {
        ScaleKind::LeaderWorkerSet(v)
    }
}
impl From<InferenceService> for ScaleKind {
    fn from(v: InferenceService) -> Self {
        ScaleKind::InferenceService(Box::new(v))
    }
}
impl From<LLMInferenceService> for ScaleKind {
    fn from(v: LLMInferenceService) -> Self {
        ScaleKind::LLMInferenceService(Box::new(v))
    }
}

impl From<ScaleKind> for ResourceKind {
    fn from(kind: ScaleKind) -> Self {
        match kind {
            ScaleKind::Deployment(_) => ResourceKind::DEPLOYMENT,
            ScaleKind::ReplicaSet(_) => ResourceKind::REPLICA_SET,
            ScaleKind::StatefulSet(_) => ResourceKind::STATEFUL_SET,
            ScaleKind::InferenceService(_) => ResourceKind::INFERENCE_SERVICE,
            ScaleKind::LLMInferenceService(_) => ResourceKind::LLM_INFERENCE_SERVICE,
            ScaleKind::Notebook(_) => ResourceKind::NOTEBOOK,
            ScaleKind::LeaderWorkerSet(_) => ResourceKind::LEADER_WORKER_SET,
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
        const LEADER_WORKER_SET = 0b100000;
        const LLM_INFERENCE_SERVICE = 0b1000000;
    }
}

/// Parse a string of resource flag characters into a [`ResourceKind`] bitflag set.
///
/// - `d` → Deployment
/// - `r` → ReplicaSet
/// - `s` → StatefulSet
/// - `i` → InferenceService
/// - `n` → Notebook
/// - `l` → LeaderWorkerSet
/// - `m` → LLMInferenceService
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
            'l' => resource_kind |= ResourceKind::LEADER_WORKER_SET,
            'm' => resource_kind |= ResourceKind::LLM_INFERENCE_SERVICE,
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

/// Why we couldn't resolve a scalable root object for a pod.
#[derive(Debug, Error)]
pub enum RootObjectError {
    /// The pod is owned by a controller that has no scale subresource by design
    /// (DaemonSets, static pods owned by a Node, etc.). These show up in GPU
    /// metrics (dcgm-exporter is the worst offender, it reports on its own idle
    /// GPUs) but are never scalable, so skipping them is expected, not a problem.
    #[error("pod {pod} is owned by non-scalable {kind}")]
    NonScalable { pod: String, kind: String },

    /// No owner ref we recognize as scalable, or no owners at all. Genuinely
    /// unexpected, worth a warning.
    #[error("no scalable root object found for pod {0}")]
    NotFound(String),

    /// A Kubernetes API call failed while walking the owner chain.
    #[error(transparent)]
    Kube(#[from] kube::Error),
}

impl TryFrom<&InstantVector> for PodMetricData {
    type Error = PodConvertError;

    fn try_from(value: &InstantVector) -> Result<Self, PodConvertError> {
        let metrics = value.metric();
        tracing::debug!("Metrics: {metrics:#?}");

        Ok(PodMetricData {
            name: metrics
                .get("exported_pod")
                .or_else(|| metrics.get("pod"))
                .ok_or_else(|| PodConvertError::UnwrapError("exported_pod/pod".into()))?
                .clone(),
            namespace: metrics
                .get("exported_namespace")
                .or_else(|| metrics.get("namespace"))
                .ok_or_else(|| PodConvertError::UnwrapError("exported_namespace/namespace".into()))?
                .clone(),
            container: metrics
                .get("exported_container")
                .or_else(|| metrics.get("container"))
                .ok_or_else(|| PodConvertError::UnwrapError("exported_container/container".into()))?
                .clone(),
            node_type: metrics
                .get("node_type")
                .cloned()
                .unwrap_or_else(|| "unknown".into()),
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
) -> anyhow::Result<(PromClient, reqwest::Client)> {
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

    let http_client = r_client.build()?;
    let prom_client = PromClient::from(http_client.clone(), url)?;

    Ok((prom_client, http_client))
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
            ScaleKind::LLMInferenceService(d) => d.$method(),
            ScaleKind::Notebook(d) => d.$method(),
            ScaleKind::LeaderWorkerSet(d) => d.$method(),
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
            ScaleKind::LeaderWorkerSet(_) => "leaderworkerset.x-k8s.io/v1".to_string(),
            ScaleKind::LLMInferenceService(_) => "serving.kserve.io/v1alpha1".to_string(),
        }
    }

    fn kind(&self) -> String {
        match self {
            ScaleKind::Deployment(_) => Deployment::KIND.to_string(),
            ScaleKind::ReplicaSet(_) => ReplicaSet::KIND.to_string(),
            ScaleKind::StatefulSet(_) => StatefulSet::KIND.to_string(),
            ScaleKind::Notebook(_) => "Notebook".to_string(),
            ScaleKind::InferenceService(_) => "InferenceService".to_string(),
            ScaleKind::LeaderWorkerSet(_) => "LeaderWorkerSet".to_string(),
            ScaleKind::LLMInferenceService(_) => "LLMInferenceService".to_string(),
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

        let result = match self {
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
            ScaleKind::LeaderWorkerSet(d) => {
                let ns = d
                    .namespace()
                    .ok_or_else(|| anyhow::anyhow!("LeaderWorkerSet has no namespace"))?;
                let api: Api<LeaderWorkerSet> = Api::namespaced(client.clone(), &ns);
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
            ScaleKind::LLMInferenceService(d) => {
                let ns = d
                    .namespace()
                    .ok_or_else(|| anyhow::anyhow!("LLMInferenceService has no namespace"))?;
                scale_llm_inference_service_to_zero(client.clone(), &d.name_unchecked(), &ns)
                    .await?;
                Ok(())
            }
        };

        if result.is_ok()
            && let Some(ns) = self.namespace()
            && let Err(e) = clear_pending_scale_at(client, &self.kind(), &self.name(), &ns).await
        {
            tracing::warn!(
                "Failed to clear pending-scale annotation for [{}] {}:{}: {e}",
                self.kind(),
                ns,
                self.name()
            );
        }

        result
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

fn workload_annotations(
    workload: &ScaleKind,
) -> Option<&std::collections::BTreeMap<String, String>> {
    match workload {
        ScaleKind::Deployment(d) => d.metadata.annotations.as_ref(),
        ScaleKind::ReplicaSet(r) => r.metadata.annotations.as_ref(),
        ScaleKind::StatefulSet(s) => s.metadata.annotations.as_ref(),
        ScaleKind::Notebook(n) => n.metadata.annotations.as_ref(),
        ScaleKind::InferenceService(i) => i.metadata.annotations.as_ref(),
        ScaleKind::LeaderWorkerSet(l) => l.metadata.annotations.as_ref(),
        ScaleKind::LLMInferenceService(l) => l.metadata.annotations.as_ref(),
    }
}

/// Read the acknowledgment annotations off a workload.
pub fn ack_status(workload: &ScaleKind) -> AckStatus {
    let not_acknowledged = AckStatus {
        acknowledged: false,
        expires_at: None,
        by_user: None,
    };

    let Some(annotations) = workload_annotations(workload) else {
        return not_acknowledged;
    };

    if let Some(expires_at_str) = annotations.get(ACK_UNTIL_ANNOTATION)
        && let Ok(expires_at) = chrono::DateTime::parse_from_rfc3339(expires_at_str)
    {
        if expires_at.timestamp() > chrono::Utc::now().timestamp() {
            return AckStatus {
                acknowledged: true,
                expires_at: Some(expires_at_str.clone()),
                by_user: annotations.get(ACK_BY_ANNOTATION).cloned(),
            };
        }
        tracing::info!(
            "Acknowledgment expired for {} in {}",
            workload.name(),
            workload.namespace().unwrap_or_default()
        );
    }

    not_acknowledged
}

/// Extract Slack mentions for a workload.
///
/// Precedence order:
/// 1. Annotation `gpu-pruner.io/slack-mentions` on the workload
/// 2. Exact namespace match in the mapper
/// 3. Longest prefix match in the mapper (keys ending in `-`)
///
/// The mention should contain space-separated Slack mention syntax:
/// - User mentions: `<@U123456789>`
/// - User group mentions: `<!subteam^S123456789>`
/// - Channel-wide mentions: `<!channel>` or `<!here>`
pub fn get_slack_mentions(
    workload: &ScaleKind,
    mapper: Option<&NamespaceMentionMapper>,
) -> Option<String> {
    if let Some(annotations) = workload_annotations(workload)
        && let Some(mentions) = annotations.get(SLACK_MENTIONS_ANNOTATION)
    {
        return Some(mentions.clone());
    }

    if let Some(mapper) = mapper
        && let Some(namespace) = workload.namespace()
        && let Some(mentions) = mapper.get_mentions(&namespace)
    {
        return Some(mentions);
    }

    None
}

/// Evaluate the ack grace period for a workload against its
/// `pending-scale-at` annotation.
pub fn check_pending_grace(
    workload: &ScaleKind,
    grace_secs: u64,
    stale_after_secs: u64,
) -> PendingScaleStatus {
    use chrono::{DateTime, Duration, Utc};

    let Some(annotations) = workload_annotations(workload) else {
        return PendingScaleStatus::NotPending;
    };

    let Some(pending_str) = annotations.get(PENDING_SCALE_ANNOTATION) else {
        return PendingScaleStatus::NotPending;
    };

    let pending_at = match DateTime::parse_from_rfc3339(pending_str) {
        Ok(dt) => dt.with_timezone(&Utc),
        Err(_) => return PendingScaleStatus::NotPending,
    };

    let now = Utc::now();
    let grace_end = pending_at + Duration::seconds(grace_secs as i64);
    let stale_at = pending_at + Duration::seconds(stale_after_secs.max(grace_secs) as i64);

    if now < grace_end {
        PendingScaleStatus::InGrace { until: grace_end }
    } else if now < stale_at {
        PendingScaleStatus::GraceExpired
    } else {
        PendingScaleStatus::Stale
    }
}

/// Apply acknowledgment annotations to a workload.
#[tracing::instrument(skip(client))]
pub async fn acknowledge_workload(
    client: KubeClient,
    kind: &str,
    name: &str,
    namespace: &str,
    duration_hours: u32,
    user: &str,
) -> anyhow::Result<()> {
    let expires_at = chrono::Utc::now() + chrono::Duration::hours(duration_hours as i64);
    let expires_at_rfc3339 = expires_at.to_rfc3339();

    let patch = serde_json::json!({
        "metadata": {
            "annotations": {
                ACK_UNTIL_ANNOTATION: expires_at_rfc3339,
                ACK_BY_ANNOTATION: user,
                PENDING_SCALE_ANNOTATION: null,
            }
        }
    });

    patch_workload(client, kind, name, namespace, &patch).await?;

    tracing::info!("Acknowledged [{kind}] {namespace}:{name} by {user} until {expires_at_rfc3339}");

    Ok(())
}

#[tracing::instrument(skip(client))]
pub async fn set_pending_scale_at(
    client: KubeClient,
    kind: &str,
    name: &str,
    namespace: &str,
) -> anyhow::Result<()> {
    let pending_at = chrono::Utc::now().to_rfc3339();
    let patch = serde_json::json!({
        "metadata": {
            "annotations": {
                PENDING_SCALE_ANNOTATION: pending_at,
            }
        }
    });
    patch_workload(client, kind, name, namespace, &patch).await
}

#[tracing::instrument(skip(client))]
pub async fn clear_pending_scale_at(
    client: KubeClient,
    kind: &str,
    name: &str,
    namespace: &str,
) -> anyhow::Result<()> {
    let patch = serde_json::json!({
        "metadata": {
            "annotations": {
                PENDING_SCALE_ANNOTATION: null,
            }
        }
    });
    patch_workload(client, kind, name, namespace, &patch).await
}

/// Dispatches a kind string to a typed `Api` bound as `$api`, then runs the
/// body. Unknown kinds return an error.
macro_rules! with_kind_api {
    ($kind:expr, $client:expr, $namespace:expr, |$api:ident| $body:expr) => {
        match $kind {
            "Deployment" => {
                let $api: Api<Deployment> = Api::namespaced($client, $namespace);
                $body
            }
            "ReplicaSet" => {
                let $api: Api<ReplicaSet> = Api::namespaced($client, $namespace);
                $body
            }
            "StatefulSet" => {
                let $api: Api<StatefulSet> = Api::namespaced($client, $namespace);
                $body
            }
            "LeaderWorkerSet" => {
                let $api: Api<LeaderWorkerSet> = Api::namespaced($client, $namespace);
                $body
            }
            "Notebook" => {
                let $api: Api<Notebook> = Api::namespaced($client, $namespace);
                $body
            }
            "InferenceService" => {
                let $api: Api<InferenceService> = Api::namespaced($client, $namespace);
                $body
            }
            "LLMInferenceService" => {
                let $api: Api<LLMInferenceService> = Api::namespaced($client, $namespace);
                $body
            }
            other => Err(anyhow::anyhow!("Unsupported resource kind: {}", other)),
        }
    };
}

#[tracing::instrument(skip(client))]
pub async fn fetch_workload(
    client: KubeClient,
    kind: &str,
    name: &str,
    namespace: &str,
) -> anyhow::Result<ScaleKind> {
    with_kind_api!(kind, client, namespace, |api| Ok(ScaleKind::from(
        api.get(name).await?
    )))
}

#[tracing::instrument(skip(client, patch))]
pub async fn patch_workload(
    client: KubeClient,
    kind: &str,
    name: &str,
    namespace: &str,
    patch: &serde_json::Value,
) -> anyhow::Result<()> {
    with_kind_api!(kind, client, namespace, |api| {
        api.patch(name, &PatchParams::default(), &Patch::Merge(patch))
            .await?;
        Ok(())
    })
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
) -> Result<ScaleKind, RootObjectError> {
    tracing::info!(
        "Finding root object of {name:?} for scale-down.",
        name = &pod_meta.name
    );
    // fast-path: LLMInferenceService pods carry standard k8s app labels
    if let Some(labels) = &pod_meta.labels
        && labels.get("app.kubernetes.io/part-of").map(|v| v.as_str())
            == Some("llminferenceservice")
        && let Some(llmis_name) = labels.get("app.kubernetes.io/name")
    {
        let namespace = pod_meta.namespace.clone().unwrap_or_default();
        let api: Api<LLMInferenceService> = Api::namespaced(client.clone(), &namespace);
        let llmis = api.get(llmis_name).await?;
        return Ok(ScaleKind::LLMInferenceService(Box::new(llmis)));
    }

    // fast-path: InferenceService pods carry a kserve-specific label
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
                    if let Ok(rs) = get_typed::<ReplicaSet>(&client, &namespace, &or.name).await {
                        if let Some(dep_or) = find_owner(&rs.metadata, "Deployment") {
                            tracing::info!("Found Deployment owning ReplicaSet!");
                            let deployment =
                                get_typed::<Deployment>(&client, &namespace, &dep_or.name).await?;

                            if let Some(llmis_or) =
                                find_owner(&deployment.metadata, "LLMInferenceService")
                            {
                                tracing::info!("Found LLMInferenceService owning Deployment!");
                                let llmis = get_typed::<LLMInferenceService>(
                                    &client,
                                    &namespace,
                                    &llmis_or.name,
                                )
                                .await?;
                                return Ok(llmis.into());
                            }

                            return Ok(deployment.into());
                        }
                        // fallthrough, replica set with no owners
                        return Ok(rs.into());
                    }
                }
                "StatefulSet" => {
                    tracing::info!("Found StatefulSet!");
                    if let Ok(ss) = get_typed::<StatefulSet>(&client, &namespace, &or.name).await {
                        if let Some(nb_or) = find_owner(&ss.metadata, "Notebook") {
                            tracing::info!("Found Notebook owning StatefulSet!");
                            let nb =
                                get_typed::<Notebook>(&client, &namespace, &nb_or.name).await?;
                            return Ok(nb.into());
                        }
                        if let Some(lws_or) = find_owner(&ss.metadata, "LeaderWorkerSet") {
                            tracing::info!("Found LeaderWorkerSet owning StatefulSet!");
                            let lws =
                                get_typed::<LeaderWorkerSet>(&client, &namespace, &lws_or.name)
                                    .await?;
                            return resolve_lws(&client, &namespace, lws).await;
                        }
                        // fallthrough, statefulset with no owners
                        return Ok(ss.into());
                    }
                }
                "LeaderWorkerSet" => {
                    tracing::info!("Found LeaderWorkerSet!");
                    if let Ok(lws) =
                        get_typed::<LeaderWorkerSet>(&client, &namespace, &or.name).await
                    {
                        return resolve_lws(&client, &namespace, lws).await;
                    }
                }
                "DaemonSet" | "Node" => {
                    return Err(RootObjectError::NonScalable {
                        pod: pod_meta.name.clone().unwrap_or_default(),
                        kind: or.kind.clone(),
                    });
                }
                kind => {
                    tracing::debug!("Ignoring unrecognized owner ref kind: {kind}");
                }
            }
        }
    }

    Err(RootObjectError::NotFound(
        pod_meta.name.clone().unwrap_or_default(),
    ))
}

fn find_owner<'a>(
    meta: &'a ObjectMeta,
    kind: &str,
) -> Option<&'a k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference> {
    meta.owner_references
        .as_ref()?
        .iter()
        .find(|or| or.kind == kind)
}

async fn get_typed<K>(client: &KubeClient, namespace: &str, name: &str) -> Result<K, kube::Error>
where
    K: kube::Resource<DynamicType = (), Scope = k8s_openapi::NamespaceResourceScope>
        + Clone
        + DeserializeOwned
        + Debug,
{
    Api::<K>::namespaced(client.clone(), namespace)
        .get(name)
        .await
}

/// A LeaderWorkerSet created by an LLMInferenceService is re-reconciled by the
/// KServe controller, so scale the owning LLMInferenceService instead.
async fn resolve_lws(
    client: &KubeClient,
    namespace: &str,
    lws: LeaderWorkerSet,
) -> Result<ScaleKind, RootObjectError> {
    if let Some(llmis_or) = find_owner(&lws.metadata, "LLMInferenceService") {
        tracing::info!("Found LLMInferenceService owning LeaderWorkerSet!");
        let llmis = get_typed::<LLMInferenceService>(client, namespace, &llmis_or.name).await?;
        return Ok(llmis.into());
    }
    Ok(lws.into())
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

/// Scale an LLMInferenceService to zero by patching spec.replicas.
///
/// The LLMInferenceService CRD does not expose a /scale subresource, so we
/// patch spec.replicas directly. In disaggregated (prefill-decode) setups,
/// prefill has its own independent replica count at spec.prefill.replicas,
/// so we zero both to avoid leaving half the pipeline running.
#[tracing::instrument(skip(client))]
async fn scale_llm_inference_service_to_zero(
    client: KubeClient,
    name: &str,
    namespace: &str,
) -> anyhow::Result<LLMInferenceService> {
    let api: Api<LLMInferenceService> = Api::namespaced(client.clone(), namespace);

    let patch = serde_json::json!({
        "spec": {
            "replicas": 0,
            "prefill": {
                "replicas": 0
            }
        }
    });

    let res = api
        .patch(name, &PatchParams::default(), &Patch::Merge(patch))
        .await?;

    Ok(res)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use k8s_openapi::api::apps::v1::{Deployment, ReplicaSet, StatefulSet};
    use kube::api::ObjectMeta;
    use resources::{
        inferenceservice::InferenceService,
        leaderworkerset::{LeaderWorkerSet, LeaderWorkerSetSpec, LeaderWorkerSetStatus},
        notebook::NotebookSpec,
    };

    use resources::llminferenceservice::LLMInferenceService;

    use crate::{
        ACK_BY_ANNOTATION, ACK_UNTIL_ANNOTATION, Meta, NamespaceMentionMapper, Notebook,
        PENDING_SCALE_ANNOTATION, PendingScaleStatus, ResourceKind, SLACK_MENTIONS_ANNOTATION,
        ScaleKind, Scaler, ack_status, check_pending_grace, get_enabled_resources,
        get_slack_mentions,
    };

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

    fn make_leader_worker_set(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        ScaleKind::LeaderWorkerSet(LeaderWorkerSet {
            metadata: ObjectMeta {
                name: Some(name.into()),
                namespace: Some(ns.into()),
                uid: uid.map(Into::into),
                ..Default::default()
            },
            spec: LeaderWorkerSetSpec {
                replicas: Some(1),
                leader_worker_template: None,
            },
            status: Some(LeaderWorkerSetStatus {
                replicas: Some(1),
                ready_replicas: Some(1),
            }),
        })
    }

    fn make_llm_inference_service(name: &str, ns: &str, uid: Option<&str>) -> ScaleKind {
        let mut llmis: LLMInferenceService = serde_json::from_value(serde_json::json!({
            "metadata": {
                "name": name,
                "namespace": ns,
            },
            "spec": {}
        }))
        .expect("valid LLMInferenceService JSON");
        llmis.metadata.uid = uid.map(Into::into);
        ScaleKind::LLMInferenceService(Box::new(llmis))
    }

    // ── get_enabled_resources ────────────────────────────────────────────

    #[test]
    fn enabled_resources_all_flags() {
        let rk = get_enabled_resources("drsinlm");
        assert!(rk.contains(ResourceKind::DEPLOYMENT));
        assert!(rk.contains(ResourceKind::REPLICA_SET));
        assert!(rk.contains(ResourceKind::STATEFUL_SET));
        assert!(rk.contains(ResourceKind::INFERENCE_SERVICE));
        assert!(rk.contains(ResourceKind::NOTEBOOK));
        assert!(rk.contains(ResourceKind::LEADER_WORKER_SET));
        assert!(rk.contains(ResourceKind::LLM_INFERENCE_SERVICE));
    }

    #[test]
    fn enabled_resources_single_llm_inference_service() {
        let rk = get_enabled_resources("m");
        assert!(rk.contains(ResourceKind::LLM_INFERENCE_SERVICE));
        assert!(!rk.contains(ResourceKind::DEPLOYMENT));
        assert!(!rk.contains(ResourceKind::INFERENCE_SERVICE));
        assert!(!rk.contains(ResourceKind::LEADER_WORKER_SET));
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
        assert!(!empty.contains(ResourceKind::LLM_INFERENCE_SERVICE));
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
    fn scale_kind_to_resource_kind_llm_inference_service() {
        let rk: ResourceKind = make_llm_inference_service("l", "ns", None).into();
        assert_eq!(rk, ResourceKind::LLM_INFERENCE_SERVICE);
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

    #[test]
    fn llm_inference_service_equality_uses_uid() {
        let a = make_llm_inference_service("llm-a", "ns", Some("uid-llm"));
        let b = make_llm_inference_service("llm-b", "ns", Some("uid-llm"));
        assert_eq!(a, b);
    }

    #[test]
    fn llm_inference_service_different_uid_not_equal() {
        let a = make_llm_inference_service("llm", "ns", Some("uid-1"));
        let b = make_llm_inference_service("llm", "ns", Some("uid-2"));
        assert_ne!(a, b);
    }

    #[test]
    fn llm_inference_service_not_equal_to_inference_service() {
        let llmis = make_llm_inference_service("x", "ns", Some("uid-1"));
        let is = make_inference_service("x", "ns", Some("uid-1"));
        assert_ne!(llmis, is);
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
        set.insert(make_llm_inference_service("l1", "ns", Some("uid-l")));
        set.insert(make_notebook("n1", "ns", Some("uid-n")));
        // duplicate of first deployment
        set.insert(make_deployment("d1", "ns", Some("uid-d")));
        assert_eq!(set.len(), 6);
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

    #[test]
    fn meta_llm_inference_service() {
        let sk = make_llm_inference_service("my-llmis", "genai", Some("llmis-uid"));
        assert_eq!(sk.name(), "my-llmis");
        assert_eq!(sk.namespace(), Some("genai".into()));
        assert_eq!(sk.kind(), "LLMInferenceService");
        assert_eq!(sk.uid(), Some("llmis-uid".into()));
        assert_eq!(sk.api_version(), "serving.kserve.io/v1alpha1");
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
    fn event_for_llm_inference_service() {
        let sk = make_llm_inference_service("my-llmis", "genai", Some("llmis-uid"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(
            event.involved_object.kind,
            Some("LLMInferenceService".into())
        );
        assert_eq!(
            event.involved_object.api_version,
            Some("serving.kserve.io/v1alpha1".into())
        );
        assert_eq!(event.involved_object.uid, Some("llmis-uid".into()));
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

    // ── LeaderWorkerSet specific tests ────────────────────────────────────

    #[test]
    fn enabled_resources_leader_worker_set_flag() {
        let rk = get_enabled_resources("l");
        assert!(rk.contains(ResourceKind::LEADER_WORKER_SET));
        assert!(!rk.contains(ResourceKind::DEPLOYMENT));
    }

    #[test]
    fn scale_kind_to_resource_kind_leader_worker_set() {
        let rk: ResourceKind = make_leader_worker_set("lws", "ns", None).into();
        assert_eq!(rk, ResourceKind::LEADER_WORKER_SET);
    }

    #[test]
    fn leader_worker_set_equality_uses_uid() {
        let a = make_leader_worker_set("lws-a", "ns", Some("uid-x"));
        let b = make_leader_worker_set("lws-b", "ns", Some("uid-x"));
        assert_eq!(a, b);
    }

    #[test]
    fn hashset_deduplicates_leader_worker_sets_by_uid() {
        let mut set = HashSet::new();
        set.insert(make_leader_worker_set("lws-1", "ns", Some("uid-lws")));
        set.insert(make_leader_worker_set("lws-2", "ns", Some("uid-lws")));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn meta_leader_worker_set() {
        let sk = make_leader_worker_set("my-lws", "training", Some("lws-uid"));
        assert_eq!(sk.name(), "my-lws");
        assert_eq!(sk.namespace(), Some("training".into()));
        assert_eq!(sk.kind(), "LeaderWorkerSet");
        assert_eq!(sk.uid(), Some("lws-uid".into()));
        assert_eq!(sk.api_version(), "leaderworkerset.x-k8s.io/v1");
    }

    // ── ack / grace period ───────────────────────────────────────────────

    fn deployment_with_annotations(annotations: &[(&str, &str)]) -> ScaleKind {
        let mut dep = Deployment::default();
        dep.metadata.name = Some("d".into());
        dep.metadata.namespace = Some("ns".into());
        dep.metadata.annotations = Some(
            annotations
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        ScaleKind::Deployment(dep)
    }

    #[test]
    fn ack_status_without_annotations_is_not_acknowledged() {
        let sk = make_deployment("d", "ns", None);
        assert!(!ack_status(&sk).acknowledged);
    }

    #[test]
    fn ack_status_honors_future_expiry() {
        let until = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let sk = deployment_with_annotations(&[
            (ACK_UNTIL_ANNOTATION, &until),
            (ACK_BY_ANNOTATION, "alice"),
        ]);
        let ack = ack_status(&sk);
        assert!(ack.acknowledged);
        assert_eq!(ack.by_user.as_deref(), Some("alice"));
    }

    #[test]
    fn ack_status_expired_is_not_acknowledged() {
        let until = (chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        let sk = deployment_with_annotations(&[(ACK_UNTIL_ANNOTATION, &until)]);
        assert!(!ack_status(&sk).acknowledged);
    }

    #[test]
    fn ack_status_garbage_timestamp_is_not_acknowledged() {
        let sk = deployment_with_annotations(&[(ACK_UNTIL_ANNOTATION, "yesterday-ish")]);
        assert!(!ack_status(&sk).acknowledged);
    }

    #[test]
    fn pending_grace_not_pending_without_annotation() {
        let sk = make_deployment("d", "ns", None);
        assert_eq!(
            check_pending_grace(&sk, 300, 900),
            PendingScaleStatus::NotPending
        );
    }

    #[test]
    fn pending_grace_in_grace_window() {
        let pending = (chrono::Utc::now() - chrono::Duration::seconds(60)).to_rfc3339();
        let sk = deployment_with_annotations(&[(PENDING_SCALE_ANNOTATION, &pending)]);
        assert!(matches!(
            check_pending_grace(&sk, 300, 900),
            PendingScaleStatus::InGrace { .. }
        ));
    }

    #[test]
    fn pending_grace_expired_within_stale_window() {
        let pending = (chrono::Utc::now() - chrono::Duration::seconds(400)).to_rfc3339();
        let sk = deployment_with_annotations(&[(PENDING_SCALE_ANNOTATION, &pending)]);
        assert_eq!(
            check_pending_grace(&sk, 300, 900),
            PendingScaleStatus::GraceExpired
        );
    }

    #[test]
    fn pending_grace_stale_after_stale_window() {
        let pending = (chrono::Utc::now() - chrono::Duration::seconds(1000)).to_rfc3339();
        let sk = deployment_with_annotations(&[(PENDING_SCALE_ANNOTATION, &pending)]);
        assert_eq!(
            check_pending_grace(&sk, 300, 900),
            PendingScaleStatus::Stale
        );
    }

    #[test]
    fn pending_grace_stale_window_never_shorter_than_grace() {
        let pending = (chrono::Utc::now() - chrono::Duration::seconds(200)).to_rfc3339();
        let sk = deployment_with_annotations(&[(PENDING_SCALE_ANNOTATION, &pending)]);
        // stale_after (100) below grace (300) is clamped up to grace
        assert!(matches!(
            check_pending_grace(&sk, 300, 100),
            PendingScaleStatus::InGrace { .. }
        ));
    }

    #[test]
    fn pending_grace_unparseable_annotation_is_not_pending() {
        let sk = deployment_with_annotations(&[(PENDING_SCALE_ANNOTATION, "not-a-time")]);
        assert_eq!(
            check_pending_grace(&sk, 300, 900),
            PendingScaleStatus::NotPending
        );
    }

    // ── slack mentions ───────────────────────────────────────────────────

    #[test]
    fn mention_mapper_exact_match_wins() {
        let mapper =
            NamespaceMentionMapper::from_json(r#"{"ml-team":"<@EXACT>","ml-":"<@PREFIX>"}"#)
                .unwrap();
        assert_eq!(mapper.get_mentions("ml-team").as_deref(), Some("<@EXACT>"));
    }

    #[test]
    fn mention_mapper_longest_prefix_wins() {
        let mapper =
            NamespaceMentionMapper::from_json(r#"{"team-":"<@TEAM>","team-ml-":"<@TEAMML>"}"#)
                .unwrap();
        assert_eq!(
            mapper.get_mentions("team-ml-dev").as_deref(),
            Some("<@TEAMML>")
        );
        assert_eq!(mapper.get_mentions("team-web").as_deref(), Some("<@TEAM>"));
    }

    #[test]
    fn mention_mapper_no_match() {
        let mapper = NamespaceMentionMapper::from_json(r#"{"alice-":"<@A>"}"#).unwrap();
        assert_eq!(mapper.get_mentions("bob-dev"), None);
    }

    #[test]
    fn mentions_annotation_beats_mapper() {
        let mapper = NamespaceMentionMapper::from_json(r#"{"ns":"<@MAPPED>"}"#).unwrap();
        let sk = deployment_with_annotations(&[(SLACK_MENTIONS_ANNOTATION, "<@ANNOTATED>")]);
        assert_eq!(
            get_slack_mentions(&sk, Some(&mapper)).as_deref(),
            Some("<@ANNOTATED>")
        );
    }

    #[test]
    fn mentions_fall_back_to_mapper() {
        let mapper = NamespaceMentionMapper::from_json(r#"{"ns":"<@MAPPED>"}"#).unwrap();
        let sk = make_deployment("d", "ns", None);
        assert_eq!(
            get_slack_mentions(&sk, Some(&mapper)).as_deref(),
            Some("<@MAPPED>")
        );
    }

    #[test]
    fn event_for_leader_worker_set() {
        let sk = make_leader_worker_set("my-lws", "ml", Some("lws-uid"));
        let event = sk.generate_scale_event().unwrap();

        assert_eq!(event.involved_object.kind, Some("LeaderWorkerSet".into()));
        assert_eq!(
            event.involved_object.api_version,
            Some("leaderworkerset.x-k8s.io/v1".into())
        );
        assert_eq!(
            event.reason,
            Some("Pod ml::my-lws was not using GPU".into())
        );
    }
}
