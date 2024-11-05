use minijinja::{context, Environment};

use once_cell::sync::Lazy;

#[cfg(feature = "otel")]
use {
    opentelemetry::global,
    opentelemetry::metrics::MetricsError,
    opentelemetry::trace::TracerProvider,
    opentelemetry::KeyValue,
    opentelemetry_otlp::{ExportConfig, WithExportConfig},
    opentelemetry_sdk::trace::Config as TraceConfig,
    opentelemetry_sdk::{runtime, Resource as OTELResource},
    tracing_opentelemetry::{MetricsLayer, OpenTelemetryLayer},
};

use resources::inferenceservice::InferenceService;
use resources::notebook::Notebook;

use std::{collections::HashSet, fmt::Debug, sync::atomic::AtomicUsize};
use tokio::{sync::mpsc::Sender, time};

use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{
    layer::{Layered, SubscriberExt},
    Layer,
};

use futures::stream::StreamExt;

use prometheus_http_query::Client;
use serde::Serialize;

use k8s_openapi::{
    api::{
        apps::v1::{Deployment, ReplicaSet, StatefulSet},
        core::v1::Pod,
    },
    chrono::{offset, Duration},
};
use kube::{api::ObjectMeta, Api, Client as KubeClient, Resource};

use clap::{Parser, ValueEnum};

use gpu_pruner::{
    get_prom_client, get_prometheus_token, Meta, PodMetricData, QueryResposne, ScaleKind, Scaler,
};

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

    /// Operation mode of the scaler process
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

    /// Log format to use
    #[clap(short, long, default_value = "pretty")]
    log_format: LogFormat,
}

#[derive(Debug, Clone, ValueEnum, Default, Serialize)]
enum Mode {
    ScaleDown,
    #[default]
    DryRun,
}

#[derive(Debug, Clone, ValueEnum, Default, Serialize)]
enum LogFormat {
    Json,
    #[default]
    Pretty,
}

static QUERY_FAILURES: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "otel")]
static RESOURCE: Lazy<OTELResource> = Lazy::new(|| {
    OTELResource::new(vec![KeyValue::new(
        opentelemetry_semantic_conventions::resource::SERVICE_NAME,
        "gpu-pruner",
    )])
});

#[cfg(feature = "otel")]
fn init_metrics() -> Result<opentelemetry_sdk::metrics::SdkMeterProvider, MetricsError> {
    let export_config = ExportConfig::default();
    opentelemetry_otlp::new_pipeline()
        .metrics(runtime::Tokio)
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_export_config(export_config),
        )
        .with_resource(RESOURCE.clone())
        .build()
}

fn setup_logging() {
    // Add a tracing filter to filter events from crates used by opentelemetry-otlp.
    // The filter levels are set as follows:
    // - Allow `info` level and above by default.
    // - Restrict `hyper`, `tonic`, and `reqwest` to `error` level logs only.
    // This ensures events generated from these crates within the OTLP Exporter are not looped back,
    // thus preventing infinite event generation.
    // Note: This will also drop events from these crates used outside the OTLP Exporter.
    // For more details, see: https://github.com/open-telemetry/opentelemetry-rust/issues/761
    let filter = EnvFilter::new("info")
        .add_directive("hyper=error".parse().unwrap())
        .add_directive("tonic=error".parse().unwrap())
        .add_directive("reqwest=error".parse().unwrap());

    let reg = tracing_subscriber::registry().with(filter);

    let args = Cli::parse();

    let json_layer = if let LogFormat::Json = args.log_format {
        Some(tracing_subscriber::fmt::layer().json())
    } else {
        None
    };

    let pretty_layer = if let LogFormat::Pretty = args.log_format {
        Some(tracing_subscriber::fmt::layer().pretty())
    } else {
        None
    };

    #[cfg(feature = "otel")]
    let metrics_layer = {
        let result = init_metrics();
        assert!(
            result.is_ok(),
            "Init metrics failed with error: {:?}",
            result.err()
        );
        let meter_provider = result.unwrap();
        global::set_meter_provider(meter_provider.clone());

        let _meter = global::meter_with_version(
            "gpu_pruner::main",
            Some("v0.2.2"),
            Some("schema_url"),
            None,
        );
        Some(MetricsLayer::new(meter_provider.clone()))
    };

    #[cfg(not(feature = "otel"))]
    let metrics_layer: Option<Box<dyn Layer<_> + Send + Sync>> = None;

    #[cfg(feature = "otel")]
    let otel_layer = {
        let provider = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(opentelemetry_otlp::new_exporter().tonic())
            .with_trace_config(TraceConfig::default().with_resource(RESOURCE.clone()))
            .install_batch(runtime::Tokio)
            .unwrap();
        global::set_tracer_provider(provider.clone());
        let trace = provider.tracer("gpu_pruner::main");
        Some(OpenTelemetryLayer::new(trace))
    };

    #[cfg(not(feature = "otel"))]
    let otel_layer: Option<Box<dyn Layer<_> + Send + Sync>> = None;

    reg.with(json_layer)
        .with(pretty_layer)
        .with(metrics_layer)
        .with(otel_layer)
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_logging();

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
                match run_query_and_scale(client, query.clone(), &args, tx.clone()).await {
                    Ok(qr) => {
                        // Reset the consecutive failure counter
                        QUERY_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(monotonic_counter.query_successes = 1, "Query succeeded");
                        tracing::info!(
                            counter.query_returned_candidates = qr.num_pods,
                            "Returned candidates"
                        )
                    }
                    Err(e) => {
                        let failures =
                            QUERY_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::error!(
                            monotonic_counter.query_failures = 1,
                            "Failed to run query and scale down! {e}"
                        );
                        if failures > 5 {
                            tracing::error!("Too many failures, exiting!");
                            break;
                        }
                    }
                }
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
            if let Err(e) = sk.scale(kube_client.clone()).await {
                tracing::error!(
                    monotonic_counter.scale_failures = 1,
                    "Failed to scale resource! {e}"
                );
                continue;
            }

            let kind = sk.kind();
            let name = sk.name();
            let namespace = sk.namespace().unwrap_or_else(|| "default".to_string());

            tracing::info!(
                monotonic_counter.scale_successes = 1,
                "Scaled Resource: [{kind}] - {namespace}:{name}",
                kind = kind,
                name = name,
                namespace = namespace
            )
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
) -> anyhow::Result<QueryResposne> {
    let response = client.query(query).get().await?;

    let data = response.data();
    let vec = data.clone().into_vector().unwrap();

    let kube_client: KubeClient = KubeClient::try_default().await?;

    let mut shutdown_events: HashSet<ScaleKind> = HashSet::new();

    for pod in &vec {
        tracing::debug!("{:#?}", pod);

        let pmd: PodMetricData = match pod.try_into() {
            Ok(pmd) => pmd,
            Err(e) => {
                tracing::error!("Failed to unwrap pod fields! {}", e);
                continue;
            }
        };

        let api = Api::<Pod>::namespaced(kube_client.clone(), &pmd.namespace);
        let pod = match api
            .get_opt(&pmd.name)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Skipping pod {namespace}:{pod_name}, retrieval error",
                    namespace = &pmd.namespace,
                    pod_name = &pmd.name
                );
                tracing::debug!("{e}");
            })
            .ok()
            .flatten()
            .or_else(|| {
                tracing::info!(
                    "Skipping pod {namespace}:{pod_name} because it no longer exists!",
                    namespace = &pmd.namespace,
                    pod_name = &pmd.name
                );
                None
            }) {
            Some(pod) => pod,
            None => continue,
        };

        if let Some(status) = pod.status.as_ref() {
            if let Some(phase) = status.phase.as_ref() {
                if phase == "Pending" {
                    tracing::info!(
                        "Skipping pod {namespace}:{pod_name}, it's still pending",
                        namespace = &pmd.namespace,
                        pod_name = &pmd.name
                    );
                    continue;
                }
            }
        };

        let create_time = pod
            .metadata
            .creation_timestamp
            .clone()
            .expect("no timestamp");

        let lookback_start = offset::Utc::now()
            - (Duration::minutes(args.duration) + Duration::seconds(args.grace_period));

        tracing::info!(
            "Pod {pod_name} create_time: {create_time} | lookback_start: {lookback_start}",
            pod_name = &pmd.name,
            create_time = create_time.0
        );
        if create_time.0 < lookback_start {
            tracing::info!("Pod older than lookback start, so eligible for scaledown.");
            let obj = find_root_object(kube_client.clone(), pod.clone().meta()).await?;
            shutdown_events.insert(obj);
        };

        let status = pod.status.unwrap().phase.unwrap().to_string();
        tracing::info!(
            "Pod [{:#?}] | CreateTime: {create_time} | Status: {status}",
            &pmd,
            create_time = create_time.0,
            status = status
        );
    }

    futures::stream::iter(shutdown_events)
        .filter_map(|obj| async {
            if let Mode::DryRun = args.run_mode {
                tracing::info!(
                    "Dry-run: Would have sent [{}] {}:{} for scaledown",
                    obj.kind(),
                    obj.namespace().unwrap_or_default(),
                    obj.name()
                );
                None // Filter out in dry-run mode
            } else {
                Some(obj) // Keep the object for sending
            }
        })
        .for_each_concurrent(None, |obj| async {
            tracing::info!(
                "Sending [{}] {}:{} for scaledown",
                obj.kind(),
                obj.namespace().unwrap_or_default(),
                obj.name()
            );

            if let Err(e) = tx.send(obj).await {
                tracing::error!("Failed to send object for scaledown: {:?}", e);
            }
        })
        .await;
    Ok(QueryResposne {
        num_pods: vec.len(),
    })
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
            let is = is_api.get(ks_label).await?;

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
