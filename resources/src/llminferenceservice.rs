use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Minimal LLMInferenceService CRD definition.
///
/// We only model `spec.replicas` because that is the sole field gpu-pruner
/// reads or patches. Everything else is captured by the `#[serde(flatten)]`
/// catch-all so the type round-trips through the API server without data loss.
///
/// The upstream CRD is still v1alpha1 and evolving rapidly; keeping this
/// hand-written (rather than kopium-generated) avoids pulling in the entire
/// PodSpec tree and makes version bumps a one-line change.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "serving.kserve.io",
    version = "v1alpha1",
    kind = "LLMInferenceService",
    plural = "llminferenceservices"
)]
#[kube(namespaced)]
pub struct LLMInferenceServiceSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicas: Option<i64>,

    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}
