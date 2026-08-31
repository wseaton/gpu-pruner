// Minimal LeaderWorkerSet CRD definition for gpu-pruner
// We only need basic structure since we use the /scale subresource

#[allow(unused_imports)]
mod prelude {
    pub use kube::CustomResource;
    pub use schemars::JsonSchema;
    pub use serde::{Deserialize, Serialize};
}
use self::prelude::*;

#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "leaderworkerset.x-k8s.io",
    version = "v1",
    kind = "LeaderWorkerSet",
    plural = "leaderworkersets"
)]
#[kube(namespaced)]
#[kube(status = "LeaderWorkerSetStatus")]
pub struct LeaderWorkerSetSpec {
    /// Number of replicas (leader-worker groups)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicas: Option<i32>,

    /// Size of each worker group
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "leaderWorkerTemplate"
    )]
    pub leader_worker_template: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
pub struct LeaderWorkerSetStatus {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicas: Option<i32>,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "readyReplicas"
    )]
    pub ready_replicas: Option<i32>,
}
