use minijinja::{context, Environment};
use resources::inferenceservice::InferenceService;
use resources::notebook::Notebook;

use secrecy::ExposeSecret;
use tokio::{sync::mpsc::Sender, time};

use std::{collections::HashSet, fmt::Debug};

use prometheus_http_query::Client;
use reqwest::header::HeaderMap;
use serde::Serialize;

use k8s_openapi::{
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        core::v1::Pod,
    },
    chrono::{offset, Duration},
};
use kube::{api::ObjectMeta, Api, Client as KubeClient, Config, Resource};

use clap::{Parser, ValueEnum};

use gpu_pruner::{Meta, ScaleKind, Scaler};

/// `gpu-pruner` is a tool to prune idle pods based on GPU utilization. It uses Prometheus to query
/// GPU utilization metrics and scales down pods that have been idle for a certain duration.
///
/// Requires a Prometheus instance to be running in the cluster w/ GPU metrics. Currently only supports
/// NVIDIA GPUs.
#[derive(Debug, Clone, Parser, Serialize)]
struct Cli {
    /// time in minutes of no gpu activity to use for pruning
    #[clap(short = 't', long, default_value = "30")]
    duration: i64,

    /// daemon mode to run in, if true, will run indefinitely
    #[clap(short, long)]
    daemon_mode: bool,

    /// interval in seconds to check for idle pods, only used in daemon mode
    #[clap(short, long, default_value = "180")]
    check_interval: u64,

    /// namespace to use for search filter, is passed down to prometheus as a pattern match
    #[clap(short, long)]
    namespace: Option<String>,

    /// Seconds of grace period to allow for metrics to be published.
    #[clap(short, long, default_value = "300")]
    grace_period: i64,

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
    /// if not provided, will try to authenticate using the service token
    /// of the currently logged in K8s user.
    #[clap(long)]
    prometheus_token: Option<String>,
}

#[derive(Debug, Clone, ValueEnum, Default, Serialize)]
enum Mode {
    ScaleDown,
    #[default]
    DryRun,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Cli::parse();

    let env: Environment = Environment::new();
    let query = env.render_str(include_str!("query.promql.j2"), context! { args })?;
    tracing::info!("Running w/ Query: {query}");

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ScaleKind>(100);

    // TODO: figure out a way to clean this up and unify the branches better
    let query_task = if args.daemon_mode {
        let args = args.clone();
        tokio::spawn(async move {
            let mut interval =
                time::interval(tokio::time::Duration::from_secs(args.check_interval));
            loop {
                interval.tick().await;
                let token = match get_prometheus_token().await {
                    Ok(token) => token,
                    Err(e) => {
                        tracing::error!("failed to get prom token: {e}");
                        panic!("failed to get prometheus  token!");
                    }
                };
                let client = get_prom_client(&args.prometheus_url, token)
                    .expect("failed to build prometheus client");
                run_query_and_scale(client, query.clone(), &args, tx.clone())
                    .await
                    .expect("failed!");
            }
        })
    } else {
        let args = args.clone();
        tokio::spawn(async move {
            let token = match get_prometheus_token().await {
                Ok(token) => token,
                Err(e) => {
                    tracing::error!("failed to get prom token: {e}");
                    panic!("failed to get prometheus  token!");
                }
            };
            let client = get_prom_client(&args.prometheus_url, token)
                .expect("failed to build prometheus client");
            run_query_and_scale(client, query, &args, tx.clone())
                .await
                .expect("failed!");
            // Explicitly drop the sender to indicate no more messages will be sent
            drop(tx);
        })
    };

    let scale_down_task = tokio::spawn(async move {
        let kube_client = KubeClient::try_default()
            .await
            .expect("failed to get kube client");

        while let Some(sk) = rx.recv().await {
            match sk.scale(kube_client.clone()).await {
                Ok(_) => {}
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

#[tracing::instrument(skip_all)]
async fn run_query_and_scale(
    client: Client,
    query: String,
    args: &Cli,
    tx: Sender<ScaleKind>,
) -> anyhow::Result<()> {
    let response = client.query(query).get().await?;

    let data = response.data();
    let vec = data.clone().into_vector().unwrap();

    let kube_client: KubeClient = KubeClient::try_default().await?;

    let mut shutdown_events: HashSet<ScaleKind> = HashSet::new();

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
            - (Duration::minutes(args.duration) + Duration::seconds(args.grace_period));

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
            "Pod {pod_name} | Namespace: {namespace} | Container: {container} | Value: {value} | NodeType: {node_type} | Age: {age} | GPU Model: {gpu_model} | Status: {status}",
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
        tracing::info!("Sending {} for scaledown: {obj:?}", obj.name());
        tx.send(obj).await?;
    }

    Ok(())
}

/// Get the token for the prometheus client
async fn get_prometheus_token() -> anyhow::Result<String> {
    if let Ok(token) = std::env::var("PROMETHEUS_TOKEN") {
        tracing::debug!("Getting token from PROMETHEUS_TOKEN");
        return Ok(token);
    }

    tracing::info!("Inferring prometheus token from K8s config");
    let config: Config = Config::infer().await?;
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

fn get_prom_client(url: &str, token: String) -> anyhow::Result<Client> {
    let mut r_client = reqwest::ClientBuilder::new();
    // add auth token as default header
    let mut header_map = HeaderMap::new();
    header_map.insert("Authorization", format!("Bearer {}", token).parse()?);

    r_client = r_client.default_headers(header_map);

    let res = Client::from(r_client.build()?, url)?;

    Ok(res)
}

/// Crawl up the owner references to find the root Deployment or StatefulSet
/// and allows an action like scaling to be performed
///
/// Deployments and StatefulSets can have multiple pods, so we shouldn't "double scale-down" them if they share a common parent and
/// both pods do not have GPU utilization. We only need to send the request once.
#[tracing::instrument(skip(client, pod_meta), fields(name = pod_meta.name))]
async fn find_root_object(client: KubeClient, pod_meta: &ObjectMeta) -> anyhow::Result<ScaleKind> {
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
            let is_api: Api<InferenceService> = Api::namespaced(client.clone(), &namespace);
            let is = is_api.get(&ks_label).await?;

            return Ok(ScaleKind::InferenceService(is));
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
                                    tracing::info!("Found Notebook owning ReplicaSet!");
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
