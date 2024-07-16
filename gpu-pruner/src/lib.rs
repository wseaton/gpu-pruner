use std::hash::{Hash, Hasher};

use k8s_openapi::{
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        autoscaling::v2::PodsMetricStatus,
        core::v1::ObjectReference,
    }, apimachinery::pkg::apis::meta::v1::MicroTime, Resource
};
use kube::{api::PostParams, client, Client, ResourceExt};
use resources::{inferenceservice::InferenceService, notebook::Notebook};
use serde::Serialize;

use minijinja::{context, Environment};

use secrecy::ExposeSecret;
use tokio::{sync::mpsc::Sender, time};
use uuid::Uuid;

use std::{collections::HashSet, fmt::Debug};

use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;

use k8s_openapi::{
    api::core::v1::{Event, Pod},
    apimachinery::pkg::apis::meta::v1::Time,
    chrono::{offset, Duration},
};
use kube::{
    api::{ObjectMeta, Patch, PatchParams},
    Api, Client as KubeClient, Config,
};

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

pub trait Meta {
    fn name(&self) -> String;
    fn namespace(&self) -> Option<String>;
    fn kind(&self) -> String;
    fn uid(&self) -> Option<String>;
}

pub trait Scaler {
    fn scale(&self, client: Client)
        -> impl std::future::Future<Output = anyhow::Result<()>> + Send;

    fn generate_scale_event(&self) -> anyhow::Result<Event>;
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

    fn kind(&self) -> String {
        match self {
            ScaleKind::Deployment(d) => Deployment::KIND.to_string(),
            ScaleKind::ReplicaSet(d) => ReplicaSet::KIND.to_string(),
            ScaleKind::StatefulSet(d) => StatefulSet::KIND.to_string(),
            ScaleKind::Notebook(d) => "Notebook".to_string(),
            ScaleKind::InferenceService(d) => "InferenceService".to_string(),
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
}

impl Scaler for ScaleKind {
    async fn scale(&self, client: Client) -> anyhow::Result<()> {
        if let Some(ns) = self.namespace() {
            let event = self.generate_scale_event()?;
            let events_api: Api<Event> = Api::namespaced(client.clone(), &ns);

            match events_api.create(&PostParams::default(), &event).await {
                Ok(_) => {
                    tracing::debug!("Emitted scale event for: {:?}", event.involved_object);
                }
                Err(e) => {
                    tracing::error!("Failed to push Event for scale down!: {e}");
                }
            };
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

    fn generate_scale_event(&self) -> anyhow::Result<Event> {
        let uuid = Uuid::new_v4();
        let now: Time = Time(offset::Utc::now());
        let event: Event = Event {
            last_timestamp: Some(now.clone()),
            first_timestamp: Some(now.clone()),
            reporting_component: Some("gpu-pruner".to_string()),
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
                name: Some(format!("gpuscaler-{}", uuid.as_simple().to_string())),
                ..Default::default()
            },
            involved_object: ObjectReference {
                api_version: None,
                field_path: None,
                kind: Some(self.kind()),
                name: Some(self.name()),
                namespace: self.namespace(),
                resource_version: None,
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

    use crate::Scaler;

    use super::{Notebook, Event, ScaleKind};

    use serde_yaml;

    #[test]
    fn make_event() {

        let sk: ScaleKind = ScaleKind::Notebook(Notebook { metadata: ObjectMeta {
            name: Some("gpu-test".to_string()),
            namespace: Some("rhoai-internal--weaton-nb".to_string()),
            ..Default::default()
        }, spec: NotebookSpec {
            template: None
        }, status: None });

        let event = sk.generate_scale_event().expect("bar");

        println!("{}", serde_yaml::to_string(&event).expect("foo"));

        assert_eq!(event.involved_object.name, Some("gpu-test".to_string()))

    }

}