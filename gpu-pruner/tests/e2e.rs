//! End-to-end tests that run against a real Kubernetes cluster (kind).
//!
//! These tests are gated behind `#[ignore]` so `cargo test` won't hit
//! the cluster unless you explicitly opt in with `cargo test -- --ignored`.
//!
//! Expected setup: a kind cluster reachable via the current kubeconfig.
//! See `just kind-create` / `just kind-delete`.

use k8s_openapi::api::{
    apps::v1::{Deployment, DeploymentSpec, StatefulSet, StatefulSetSpec},
    core::v1::{
        Container, Event, Namespace, PodSpec, PodTemplateSpec, Service, ServicePort, ServiceSpec,
    },
};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::LabelSelector;
use kube::{
    Api, Client, ResourceExt,
    api::{DeleteParams, ObjectMeta, PostParams},
};
use std::collections::BTreeMap;

use gpu_pruner::{Meta, ScaleKind, Scaler, find_root_object};

/// Per-test namespace so tests can run in parallel without stomping each other.
async fn create_test_namespace(client: &Client, name: &str) -> String {
    let ns_api: Api<Namespace> = Api::all(client.clone());
    let ns = Namespace {
        metadata: ObjectMeta {
            name: Some(name.into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let _ = ns_api.create(&PostParams::default(), &ns).await;
    name.to_string()
}

async fn delete_test_namespace(client: &Client, name: &str) {
    let ns_api: Api<Namespace> = Api::all(client.clone());
    let _ = ns_api.delete(name, &DeleteParams::default()).await;
}

fn test_labels() -> BTreeMap<String, String> {
    BTreeMap::from([("app".into(), "gpu-pruner-e2e".into())])
}

/// Wait for a deployment to have at least one ready pod.
async fn wait_for_deployment_ready(api: &Api<Deployment>, name: &str) {
    for _ in 0..60 {
        if let Ok(dep) = api.get(name).await
            && let Some(status) = dep.status
            && status.ready_replicas.unwrap_or(0) > 0
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    panic!("deployment {name} never became ready");
}

/// Wait for a statefulset to have at least one ready pod.
async fn wait_for_statefulset_ready(api: &Api<StatefulSet>, name: &str) {
    for _ in 0..60 {
        if let Ok(ss) = api.get(name).await
            && let Some(status) = ss.status
            && status.ready_replicas.unwrap_or(0) > 0
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    panic!("statefulset {name} never became ready");
}

fn make_deployment(name: &str, ns: &str) -> Deployment {
    let labels = test_labels();
    Deployment {
        metadata: ObjectMeta {
            name: Some(name.into()),
            namespace: Some(ns.into()),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(1),
            selector: LabelSelector {
                match_labels: Some(labels.clone()),
                ..Default::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels),
                    ..Default::default()
                }),
                spec: Some(PodSpec {
                    containers: vec![Container {
                        name: "pause".into(),
                        image: Some("registry.k8s.io/pause:3.10".into()),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_statefulset(name: &str, ns: &str) -> StatefulSet {
    let labels = test_labels();
    StatefulSet {
        metadata: ObjectMeta {
            name: Some(name.into()),
            namespace: Some(ns.into()),
            ..Default::default()
        },
        spec: Some(StatefulSetSpec {
            replicas: Some(1),
            selector: LabelSelector {
                match_labels: Some(labels.clone()),
                ..Default::default()
            },
            service_name: Some(format!("{name}-svc")),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels),
                    ..Default::default()
                }),
                spec: Some(PodSpec {
                    containers: vec![Container {
                        name: "pause".into(),
                        image: Some("registry.k8s.io/pause:3.10".into()),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
            },
            // no PVCs needed for this test
            volume_claim_templates: Some(vec![]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_headless_service(name: &str, ns: &str) -> Service {
    Service {
        metadata: ObjectMeta {
            name: Some(name.into()),
            namespace: Some(ns.into()),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()),
            selector: Some(test_labels()),
            ports: Some(vec![ServicePort {
                port: 80,
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

// ── find_root_object tests ───────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn find_root_object_deployment_chain() {
    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-dep").await;

    let dep_api: Api<Deployment> = Api::namespaced(client.clone(), &ns);
    let dep = dep_api
        .create(&PostParams::default(), &make_deployment("e2e-dep", &ns))
        .await
        .unwrap();

    wait_for_deployment_ready(&dep_api, "e2e-dep").await;

    // grab a pod owned by this deployment
    let pod_api: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(client.clone(), &ns);
    let pods = pod_api.list(&Default::default()).await.unwrap();
    let pod = pods.items.first().expect("no pods found for deployment");

    let root = find_root_object(client.clone(), &pod.metadata)
        .await
        .unwrap();

    // should resolve to the Deployment, not the intermediate ReplicaSet
    assert_eq!(root.kind(), "Deployment");
    assert_eq!(root.name(), dep.name_unchecked());
    assert_eq!(root.namespace(), Some(ns.clone()));

    delete_test_namespace(&client, &ns).await;
}

#[tokio::test]
#[ignore]
async fn find_root_object_statefulset() {
    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-ss").await;

    // statefulsets need a headless service
    let svc_api: Api<Service> = Api::namespaced(client.clone(), &ns);
    svc_api
        .create(
            &PostParams::default(),
            &make_headless_service("e2e-ss-svc", &ns),
        )
        .await
        .unwrap();

    let ss_api: Api<StatefulSet> = Api::namespaced(client.clone(), &ns);
    let ss = ss_api
        .create(&PostParams::default(), &make_statefulset("e2e-ss", &ns))
        .await
        .unwrap();

    wait_for_statefulset_ready(&ss_api, "e2e-ss").await;

    let pod_api: Api<k8s_openapi::api::core::v1::Pod> = Api::namespaced(client.clone(), &ns);
    let pods = pod_api.list(&Default::default()).await.unwrap();
    let pod = pods.items.first().expect("no pods found for statefulset");

    let root = find_root_object(client.clone(), &pod.metadata)
        .await
        .unwrap();

    // no Notebook owner, so should resolve to the StatefulSet itself
    assert_eq!(root.kind(), "StatefulSet");
    assert_eq!(root.name(), ss.name_unchecked());

    delete_test_namespace(&client, &ns).await;
}

#[tokio::test]
#[ignore]
async fn find_root_object_no_owner_refs_errors() {
    let client = Client::try_default().await.unwrap();

    // a bare pod with no owner references should fail
    let meta = ObjectMeta {
        name: Some("orphan-pod".into()),
        namespace: Some("default".into()),
        ..Default::default()
    };

    let result = find_root_object(client.clone(), &meta).await;
    assert!(result.is_err());
}

// ── scale_to_zero tests ─────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn scale_deployment_to_zero() {
    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-scale-dep").await;

    let dep_api: Api<Deployment> = Api::namespaced(client.clone(), &ns);
    dep_api
        .create(&PostParams::default(), &make_deployment("e2e-scale", &ns))
        .await
        .unwrap();
    wait_for_deployment_ready(&dep_api, "e2e-scale").await;

    let dep = dep_api.get("e2e-scale").await.unwrap();
    let sk = ScaleKind::Deployment(dep);
    sk.scale(client.clone()).await.unwrap();

    // verify it scaled to zero
    let dep = dep_api.get("e2e-scale").await.unwrap();
    let replicas = dep.spec.unwrap().replicas.unwrap_or(1);
    assert_eq!(replicas, 0, "deployment should be scaled to 0 replicas");

    // verify an event was created
    let events_api: Api<Event> = Api::namespaced(client.clone(), &ns);
    let events = events_api.list(&Default::default()).await.unwrap();
    let scale_events: Vec<_> = events
        .items
        .iter()
        .filter(|e| {
            e.metadata
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("gpuscaler-"))
        })
        .collect();
    assert!(
        !scale_events.is_empty(),
        "expected at least one gpuscaler event"
    );

    delete_test_namespace(&client, &ns).await;
}

#[tokio::test]
#[ignore]
async fn scale_statefulset_to_zero() {
    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-scale-ss").await;

    let svc_api: Api<Service> = Api::namespaced(client.clone(), &ns);
    svc_api
        .create(
            &PostParams::default(),
            &make_headless_service("e2e-scale-ss-svc", &ns),
        )
        .await
        .unwrap();

    let ss_api: Api<StatefulSet> = Api::namespaced(client.clone(), &ns);
    ss_api
        .create(
            &PostParams::default(),
            &make_statefulset("e2e-scale-ss", &ns),
        )
        .await
        .unwrap();
    wait_for_statefulset_ready(&ss_api, "e2e-scale-ss").await;

    let ss = ss_api.get("e2e-scale-ss").await.unwrap();
    let sk = ScaleKind::StatefulSet(ss);
    sk.scale(client.clone()).await.unwrap();

    let ss = ss_api.get("e2e-scale-ss").await.unwrap();
    let replicas = ss.spec.unwrap().replicas.unwrap_or(1);
    assert_eq!(replicas, 0, "statefulset should be scaled to 0 replicas");

    delete_test_namespace(&client, &ns).await;
}

// ── event generation against real cluster ────────────────────────────

#[tokio::test]
#[ignore]
async fn event_posted_to_cluster() {
    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-event").await;

    let dep_api: Api<Deployment> = Api::namespaced(client.clone(), &ns);
    dep_api
        .create(&PostParams::default(), &make_deployment("e2e-event", &ns))
        .await
        .unwrap();
    wait_for_deployment_ready(&dep_api, "e2e-event").await;

    let dep = dep_api.get("e2e-event").await.unwrap();
    let sk = ScaleKind::Deployment(dep);
    let event = sk.generate_scale_event().unwrap();

    let events_api: Api<Event> = Api::namespaced(client.clone(), &ns);
    let created = events_api
        .create(&PostParams::default(), &event)
        .await
        .unwrap();

    assert!(created.metadata.name.unwrap().starts_with("gpuscaler-"));
    assert_eq!(created.action, Some("scale_down".into()));
    assert_eq!(created.involved_object.kind, Some("Deployment".into()));
    assert_eq!(created.involved_object.name, Some("e2e-event".into()));

    delete_test_namespace(&client, &ns).await;
}

// ── dedup in HashSet with real cluster UIDs ──────────────────────────

#[tokio::test]
#[ignore]
async fn hashset_dedup_with_real_uids() {
    use std::collections::HashSet;

    let client = Client::try_default().await.unwrap();
    let ns = create_test_namespace(&client, "gpu-pruner-e2e-dedup").await;

    let dep_api: Api<Deployment> = Api::namespaced(client.clone(), &ns);
    dep_api
        .create(&PostParams::default(), &make_deployment("e2e-dedup", &ns))
        .await
        .unwrap();
    wait_for_deployment_ready(&dep_api, "e2e-dedup").await;

    let dep = dep_api.get("e2e-dedup").await.unwrap();

    let mut set = HashSet::new();
    // inserting the same deployment twice should dedup
    set.insert(ScaleKind::Deployment(dep.clone()));
    set.insert(ScaleKind::Deployment(dep));
    assert_eq!(set.len(), 1, "same deployment should dedup in HashSet");

    delete_test_namespace(&client, &ns).await;
}
