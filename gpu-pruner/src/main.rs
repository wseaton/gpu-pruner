use minijinja::{context, Environment};
use resources::inferenceservice::InferenceService;
use resources::notebook::Notebook;

use tokio::{sync::mpsc::Sender, time};

use std::{collections::HashSet, fmt::Debug};

use prometheus_http_query::Client;
use reqwest::header::HeaderMap;
use serde::{de::DeserializeOwned, Serialize};

use k8s_openapi::{
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        core::v1::Pod,
    },
    chrono::{offset, Duration},
};
use kube::{
    api::{ObjectMeta, Patch, PatchParams},
    Api, Client as KubeClient, Resource, 
};

use clap::{Parser, ValueEnum};

/// Seconds of grace period to allow for metrics to be published.
const GRACE_PERIOD: i64 = 5 * 60;

/// `gpu-pruner` is a tool to prune idle pods based on GPU utilization. It uses Prometheus to query
/// GPU utilization metrics and scales down pods that have been idle for a certain duration.
///
/// Requires a Prometheus instance to be running in the cluster w/ GPU metrics. Currently only supports
/// NVIDIA GPUs.
#[derive(Debug, Clone, Parser, Serialize)]
struct Cli {
    /// time in minutes of no gpu activity to use for pruning
    #[clap(short = 't', long, default_value = "30")]
    duration: u64,

    /// daemon mode to run in, if true, will run indefinitely
    #[clap(short, long)]
    daemon_mode: bool,

    /// interval in seconds to check for idle pods, only used in daemon mode
    #[clap(short, long, default_value = "180")]
    check_interval: u64,

    /// namespace to use for search filter, is passed down to prometheus as a pattern match
    #[clap(short, long)]
    namespace: Option<String>,

    /// model name of GPU to use for filter, eg. "NVIDIA A10G", is passed down to prometheus as a pattern match
    #[clap(short, long)]
    model_name: Option<String>,

    /// Operation mode, either "dry-run" or "scale-down".
    #[clap(short, long, default_value = "dry-run")]
    run_mode: Mode,

    /// Prometheus URL to query for GPU metrics
    /// eg. "http://prometheus-k8s.openshift-monitoring.svc:9090"
    #[clap(long)]
    prometheus_url: String,

    /// Prometheus token to use for authentication,
    /// if not provided, will try to authenticate using the service account token
    /// for the currently logged in OpenShift user.
    #[clap(long)]
    prometheus_token: Option<String>,
}

#[derive(Debug, Clone, ValueEnum, Default, Serialize)]
enum Mode {
    ScaleDown,
    #[default]
    DryRun,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
struct ScaleResource {
    kind: ScaleKind,
    name: String,
    namespace: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
enum ScaleKind {
    Deployment,
    ReplicaSet,
    StatefulSet,
    InferenceService,
    Notebook,
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = Cli::parse();


    if let Ok(token) = std::env::var("PROMETHEUS_TOKEN") {
        tracing::info!("Getting token from PROMETHEUS_TOKEN");
        args.prometheus_token = Some(token);
    };


    let env: Environment = Environment::new();
    let query = env.render_str(include_str!("query.promql.j2"), context! { args })?;

    let client = get_prom_client(&args)?;
    tracing::info!("Running w/ Query: {query}");


    let (tx, mut rx) = tokio::sync::mpsc::channel::<ScaleResource>(100);

    let query_task = if args.daemon_mode {
        let args = args.clone();
        tokio::spawn(async move {
            let mut interval =
                time::interval(tokio::time::Duration::from_secs(args.check_interval));
            loop {
                interval.tick().await;
                run_query_and_scale(client.clone(), query.clone(), &args, tx.clone())
                    .await
                    .expect("failed!");
            }
        })
    } else {
        let args = args.clone();
        tokio::spawn(async move {
            run_query_and_scale(client, query, &args, tx.clone())
                .await
                .expect("failed!");
            // Explicitly drop the sender to indicate no more messages will be sent
            drop(tx);
        })
    };

    let scale_down_task = tokio::spawn(async move {
        let kube_client = KubeClient::try_default().await.expect("failed to get kube client");

        while let Some(sr) = rx.recv().await {
            match scale_resources(args.clone(), sr, kube_client.clone()).await {
                Ok(_) => {},
                Err(e) => {
                    tracing::error!("Failed to scale resource! {e}");
                }
            }
        }
    });

    _ = tokio::try_join! {
        query_task,
        scale_down_task
    }?;

    Ok(())
}

async fn scale_resources(
    args: Cli,
    sr: ScaleResource,
    kube_client: KubeClient,
) -> anyhow::Result<()> {
    match args.run_mode {
        Mode::ScaleDown => {
                tracing::info!("Scale down: {:?}", sr);
                match sr.kind {
                    ScaleKind::InferenceService => {
                        let _ = scale_inference_service_to_zero(
                            kube_client.clone(),
                            &sr.name,
                            &sr.namespace,
                        )
                        .await?;
                    }
                    ScaleKind::Notebook => {
                        let _ = scale_notebook_to_zero(kube_client.clone(), &sr.name, &sr.namespace).await?;
                    },
                    ScaleKind::Deployment => {
                        let deployments: Api<Deployment> = Api::namespaced(kube_client.clone(), &sr.namespace);
                        scale_to_zero(
                            deployments,
                            &sr.name,
                        )
                        .await?;
                    },
                    ScaleKind::ReplicaSet => {
                        let replica_sets: Api<ReplicaSet> = Api::namespaced(kube_client.clone(), &sr.namespace);
                        scale_to_zero(
                            replica_sets,
                            &sr.name,
                        )
                        .await?;
                    },
                    ScaleKind::StatefulSet => {
                        let stateful_sets: Api<StatefulSet> = Api::namespaced(kube_client.clone(), &sr.namespace);
                        scale_to_zero(
                            stateful_sets,
                            &sr.name,
                        )
                        .await?;
                    },
                }
        }
        Mode::DryRun => {
                tracing::info!("Would scale down: {:?}", sr);
        }
    }
    Ok(())
}

async fn run_query_and_scale(
    client: Client,
    query: String,
    args: &Cli,
    tx: Sender<ScaleResource>,
) -> anyhow::Result<()> {
    let response = client.query(query).get().await?;

    let data = response.data();
    let vec = data.clone().into_vector().unwrap();

    let kube_client: KubeClient = KubeClient::try_default().await?;

    let mut shutdown_events: HashSet<ScaleResource> = HashSet::new();

    for pod in vec {
        let pod_name = pod.metric().get("exported_pod").unwrap();
        let namespace = pod.metric().get("exported_namespace").unwrap();
        let container = pod.metric().get("exported_container").unwrap();
        let node_type = pod.metric().get("node_type").unwrap();
        let gpu_model = pod
            .metric()
            .get("modelName")
            .map(|model| model.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let value = pod.sample().value();

        let api = Api::<Pod>::namespaced(kube_client.clone(), namespace);
        let maybe_pod = match api.get_opt(pod_name).await {
            Ok(pod) => pod,
            Err(e) => {
                tracing::error!("Skipping pod {pod_name}, retrieval error");
                tracing::debug!("{e}");
                continue;
            }
        };
        let pod = match maybe_pod {
            Some(pod) => pod,
            None => {
                tracing::info!("Skipping pod {pod_name} because it no longer exists!");
                continue;
            }
        };


        if let Some(status) = pod.status.as_ref() {
            if let Some(phase) = status.phase.as_ref() {
                if phase == "Pending" {
                    tracing::info!("Skipping pod {pod_name}, it's still pending");
                    continue;
                }
            }
        };

        let age = pod.metadata.creation_timestamp.clone().unwrap();

        let lookback_start = offset::Utc::now()
            - (Duration::minutes(args.duration as i64) + Duration::seconds(GRACE_PERIOD));

        tracing::info!(
            "Pod {pod_name} age: {age} | lookback_start: {lookback_start}",
            pod_name = pod_name,
            age = age.0
        );
        if age.0 < lookback_start {
            tracing::info!("Pod older than lookback start, so eligible for scaledown.");
            let obj = find_root_object(kube_client.clone(), pod.clone().meta()).await?;
            shutdown_events.insert(obj);
        };

        let status = pod.status.unwrap().phase.unwrap().to_string();
        tracing::info!(
            "Pod: {pod_name} | Namespace: {namespace} | Container: {container} | Value: {value} | NodeType: {node_type} | Age: {age} | GPU Model: {gpu_model} | Status: {status}",
            pod_name = pod_name,
            namespace = namespace,
            container = container,
            value = value,
            node_type = node_type,
            age = age.0,
            gpu_model = gpu_model,
            status = status
        );
    }

    for obj in shutdown_events {
        tracing::info!("Sending {} for scaledown: {obj:?}", obj.name);
        tx.send(obj).await?;
    }

    Ok(())
}

fn get_prom_client(cli: &Cli) -> anyhow::Result<Client> {
    // get a token by running oc whomai -t via shell
    tracing::info!("Getting token via `oc whoami -t`");
    let token = match cli.prometheus_token.as_ref() {
        Some(token) => token.to_string(),
        None => {
            tracing::info!("No token provided, trying to get token from oc");
            let token = std::process::Command::new("oc")
                .args(["whoami", "-t"])
                .output()?
                .stdout;
            std::str::from_utf8(&token)?.trim().to_string()
        }
    };

    let mut r_client = reqwest::ClientBuilder::new();
    // add auth token as default header
    let mut header_map = HeaderMap::new();
    header_map.insert("Authorization", format!("Bearer {}", token).parse()?);

    r_client = r_client.default_headers(header_map);

    let res = Client::from(r_client.build()?, &cli.prometheus_url)?;

    Ok(res)
}

/// Crawl up the owner references to find the root Deployment or StatefulSet
/// and allows an action like scaling to be performed
///
/// Deployments and StatefulSets can have multiple pods, so we shouldn't "double scale-down" them if they share a common parent and
/// both pods do not have GPU utilization. We only need to send the request once.
#[tracing::instrument(skip(client, pod_meta), fields(name = pod_meta.name))]
async fn find_root_object(
    client: KubeClient,
    pod_meta: &ObjectMeta,
) -> anyhow::Result<ScaleResource> {
    tracing::info!(
        "Finding root object of {name:?} for scale-down.",
        name = &pod_meta.name
    );
    // first, check for the special kserve label
    // if it exists, we can go directly to the InferenceService
    // and scale it down
    if let Some(labels) = &pod_meta.labels {
        if let Some(ks_label) = labels.get("serving.kserve.io/inferenceservice") {
            let namespace = pod_meta.namespace.clone().unwrap_or_default();
            return Ok(ScaleResource {
                    kind: ScaleKind::InferenceService,
                    name: ks_label.clone(),
                    namespace: namespace.clone(),
                });
        }
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
                                    return Ok(ScaleResource {
                                            kind: ScaleKind::Deployment,
                                            name: rs_or.name.clone(),
                                            namespace: namespace.clone(),
                                        });
                                }
                            }
                        }
                        // fallthrough, replica set with no owners
                        return Ok(ScaleResource {
                            kind: ScaleKind::ReplicaSet,
                            name: or.name.clone(),
                            namespace: namespace.clone(),
                        });
                    }
                }
                "StatefulSet" => {
                    tracing::info!("Found StatefulSet!");
                    let ss_api: Api<StatefulSet> = Api::namespaced(client.clone(), &namespace);
                    if let Ok(ss) = ss_api.get(&or.name).await {
                        if let Some(ss_meta) = ss.metadata.owner_references.as_ref() {
                            for ss_or in ss_meta {
                                if ss_or.kind == "Notebook" {
                                    tracing::info!("Found Notebook owning ReplicaSet!");
                                    return Ok(ScaleResource {
                                            kind: ScaleKind::Notebook,
                                            name: ss_or.name.clone(),
                                            namespace: namespace.clone(),
                                        });
                                }
                            }
                        }
                        // fallthrough, statefulset with no owners
                        return Ok(ScaleResource {
                                kind: ScaleKind::StatefulSet,
                                name: or.name.clone(),
                                namespace: namespace.clone(),
                            });
                    }
                }
                "Deployment" => {
                    tracing::info!("Found Deployment!");
                    return Ok(ScaleResource {
                            kind: ScaleKind::Deployment,
                            name: or.name.clone(),
                            namespace: namespace.clone(),
                        });
                }
                _ => {
                    tracing::warn!("Found no ORs!")
                }
            }
        }
    }

    Err(anyhow::anyhow!("oops, nothing found!"))
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

    // by setting spec.predictor.minReplicas to 0 we allow Kserve to scale down the request for us, it will get
    // rescaled automatically when traffic kicks back up. the user will need to manually set minRepliacs
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
