use minijinja::{Environment, context};

#[cfg(feature = "otel")]
use std::sync::LazyLock;
#[cfg(feature = "otel")]
use {
    opentelemetry::global,
    opentelemetry::trace::TracerProvider,
    opentelemetry_otlp::{MetricExporter, SpanExporter},
    opentelemetry_sdk::Resource as OTELResource,
    opentelemetry_sdk::metrics::SdkMeterProvider,
    opentelemetry_sdk::trace::SdkTracerProvider,
    tracing_opentelemetry::{MetricsLayer, OpenTelemetryLayer},
};

use std::{collections::HashSet, fmt::Debug, sync::atomic::AtomicUsize};
use tokio::{sync::mpsc::Sender, time};

use tracing_subscriber::EnvFilter;
#[cfg(not(feature = "otel"))]
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use futures::stream::StreamExt;

use prometheus_http_query::Client;
use serde::Serialize;

use jiff::{SignedDuration, Timestamp};
use k8s_openapi::api::core::v1::Pod;
use kube::{Api, Client as KubeClient, Resource};

use clap::{Parser, ValueEnum};

use gpu_pruner::{
    Meta, PodMetricData, QueryResponse, ScaleKind, Scaler, TlsMode, find_root_object,
    get_enabled_resources, get_prom_client, get_prometheus_token,
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

    /// Specifcy enabled resources with a string of letters
    ///
    /// - `d` for Deployment
    /// - `r` for ReplicaSet
    /// - `s` for StatefulSet
    /// - `i` for InferenceService
    /// - `n` for Notebook
    #[clap(short, long, default_value = "drsin")]
    enabled_resources: String,

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

    /// Power draw threshold in watts. When set, GPUs showing peak power usage above this value
    /// over the lookback window are excluded from idle candidates even if compute utilization is zero.
    /// Useful as a corroborating signal (e.g. 100 for A10G, 150 for A100/H100).
    #[clap(long)]
    power_threshold: Option<f64>,

    /// Set when the Prometheus ServiceMonitor uses honorLabels: true.
    /// Controls whether the query uses native DCGM label names (pod/namespace/container)
    /// or the Prometheus-prefixed names (exported_pod/exported_namespace/exported_container).
    #[clap(long, default_value = "false")]
    honor_labels: bool,

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

    #[clap(long, default_value = "verify")]
    prometheus_tls_mode: TlsMode,

    /// Custom .crt file to use for TLS verification
    #[clap(long)]
    prometheus_tls_cert: Option<String>,

    /// Log format to use
    #[clap(short, long, default_value = "default")]
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
    Default,
    Pretty,
}

static QUERY_FAILURES: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "otel")]
static RESOURCE: LazyLock<OTELResource> = LazyLock::new(|| {
    OTELResource::builder()
        .with_service_name("gpu-pruner")
        .build()
});

#[cfg(feature = "otel")]
fn init_metrics() -> anyhow::Result<SdkMeterProvider> {
    let exporter = MetricExporter::builder().with_tonic().build()?;

    let provider = SdkMeterProvider::builder()
        .with_periodic_exporter(exporter)
        .with_resource(RESOURCE.clone())
        .build();

    Ok(provider)
}

fn setup_logging(args: &Cli) -> OtelGuard {
    // Add a tracing filter to filter events from crates used by opentelemetry-otlp.
    // The filter levels are set as follows:
    // - Allow `info` level and above by default.
    // - Restrict `hyper`, `tonic`, and `reqwest` to `error` level logs only.
    // This ensures events generated from these crates within the OTLP Exporter are not looped back,
    // thus preventing infinite event generation.
    // Note: This will also drop events from these crates used outside the OTLP Exporter.
    // For more details, see: https://github.com/open-telemetry/opentelemetry-rust/issues/761
    #[cfg(feature = "otel")]
    let filter = EnvFilter::from_default_env()
        .add_directive("hyper=error".parse().unwrap())
        .add_directive("tonic=error".parse().unwrap())
        .add_directive("reqwest=error".parse().unwrap());

    #[cfg(not(feature = "otel"))]
    let filter = EnvFilter::from_default_env();
    let reg = tracing_subscriber::registry().with(filter);

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

    let default_layer = if let LogFormat::Default = args.log_format {
        Some(tracing_subscriber::fmt::layer())
    } else {
        None
    };

    #[cfg(feature = "otel")]
    let meter_provider = get_meter_provider();

    #[cfg(feature = "otel")]
    let metrics_layer = {
        let _meter = global::meter("gpu_pruner::main");
        Some(MetricsLayer::new(meter_provider.clone()))
    };

    #[cfg(not(feature = "otel"))]
    let metrics_layer: Option<Box<dyn Layer<_> + Send + Sync>> = None;

    #[cfg(feature = "otel")]
    let (otel_layer, tracer_provider) = {
        let exporter = SpanExporter::builder()
            .with_tonic()
            .build()
            .expect("failed to create span exporter");

        let provider = SdkTracerProvider::builder()
            .with_resource(RESOURCE.clone())
            .with_batch_exporter(exporter)
            .build();

        global::set_tracer_provider(provider.clone());
        let tracer = provider.tracer("gpu_pruner::main");
        (Some(OpenTelemetryLayer::new(tracer)), provider)
    };

    #[cfg(not(feature = "otel"))]
    let otel_layer: Option<Box<dyn Layer<_> + Send + Sync>> = None;

    reg.with(json_layer)
        .with(default_layer)
        .with(pretty_layer)
        .with(metrics_layer)
        .with(otel_layer)
        .init();

    #[cfg(feature = "otel")]
    {
        OtelGuard {
            meter_provider,
            tracer_provider,
        }
    }

    #[cfg(not(feature = "otel"))]
    OtelGuard
}

#[cfg(feature = "otel")]
fn get_meter_provider() -> SdkMeterProvider {
    let meter_provider = init_metrics().expect("failed to init metrics");
    global::set_meter_provider(meter_provider.clone());
    meter_provider
}

#[cfg(feature = "otel")]
struct OtelGuard {
    meter_provider: SdkMeterProvider,
    tracer_provider: SdkTracerProvider,
}

#[cfg(not(feature = "otel"))]
struct OtelGuard;

#[cfg(feature = "otel")]
impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(err) = self.tracer_provider.shutdown() {
            eprintln!("{err:?}");
        }
        if let Err(err) = self.meter_provider.shutdown() {
            eprintln!("{err:?}");
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let _guard = setup_logging(&args);
    let enabled_resources = get_enabled_resources(&args.enabled_resources);
    tracing::info!("Enabled resources: {enabled_resources:?}");

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
                let client = get_prom_client(
                    &args.prometheus_url,
                    token,
                    args.prometheus_tls_mode,
                    args.prometheus_tls_cert.clone(),
                )
                .expect("failed to build prometheus client");
                match run_query_and_scale(client, query.clone(), &args, tx.clone()).await {
                    Ok(qr) => {
                        // Reset the consecutive failure counter
                        QUERY_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!(monotonic_counter.query_successes = 1, "Query succeeded");
                        tracing::info!(
                            counter.query_returned_candidates = qr.num_pods,
                            "Returned candidates"
                        );
                        tracing::info!(
                            counter.query_returned_shutdown_events = qr.shutdown_events,
                            "Returned shutdown events"
                        );
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
            let client = get_prom_client(
                &args.prometheus_url,
                token,
                args.prometheus_tls_mode,
                args.prometheus_tls_cert.clone(),
            )
            .expect("failed to build prometheus client");
            match run_query_and_scale(client, query, &args, tx.clone()).await {
                Ok(qr) => {
                    tracing::info!(monotonic_counter.query_successes = 1, "Query succeeded");
                    tracing::info!(
                        counter.query_returned_candidates = qr.num_pods,
                        "Returned candidates"
                    );
                    tracing::info!(
                        counter.query_returned_shutdown_events = qr.shutdown_events,
                        "Returned shutdown events"
                    );
                }
                Err(e) => tracing::error!("Failed to scale down! {e}"),
            }
            // Explicitly drop the sender to indicate no more messages will be sent
            drop(tx);
        })
    };

    let scale_down_task = tokio::spawn(async move {
        let kube_client = KubeClient::try_default()
            .await
            .expect("failed to get kube client");

        while let Some(sk) = rx.recv().await {
            // Check if the resource is enabled
            if !enabled_resources.contains(sk.clone().into()) {
                tracing::info!(
                    "Skipping resource type {kind:?} because it is not enabled",
                    kind = sk.kind()
                );
                continue;
            }

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
) -> anyhow::Result<QueryResponse> {
    let response = match client.query(query).get().await {
        Ok(response) => response,
        Err(e) => {
            tracing::error!("Failed to run query! {e}");
            return Err(anyhow::anyhow!("Failed to run query! {e}"));
        }
    };

    let vec = response
        .data()
        .clone()
        .into_vector()
        .expect("expected vector response from prometheus");

    let kube_client: KubeClient = KubeClient::try_default().await?;

    let lookback_duration =
        SignedDuration::from_mins(args.duration) + SignedDuration::from_secs(args.grace_period);

    // Process pods concurrently (up to 10 at a time) instead of serially.
    // Each pod requires 1-3 API calls (get pod, walk owner refs), so parallelism
    // cuts wall-clock time significantly on large result sets.
    let num_pods = vec.len();
    let results: Vec<Option<ScaleKind>> = futures::stream::iter(vec)
        .map(|pod| {
            let kube_client = kube_client.clone();
            async move {
                tracing::debug!("{:#?}", pod);

                let pmd: PodMetricData = match (&pod).try_into() {
                    Ok(pmd) => pmd,
                    Err(e) => {
                        tracing::error!("Failed to unwrap pod fields! {}", e);
                        return None;
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
                {
                    Some(pod) => pod,
                    None => {
                        tracing::info!(
                            "Skipping pod {namespace}:{pod_name} because it no longer exists!",
                            namespace = &pmd.namespace,
                            pod_name = &pmd.name
                        );
                        return None;
                    }
                };

                if let Some(status) = pod.status.as_ref()
                    && let Some(phase) = status.phase.as_ref()
                    && phase == "Pending"
                {
                    tracing::info!(
                        "Skipping pod {namespace}:{pod_name}, it's still pending",
                        namespace = &pmd.namespace,
                        pod_name = &pmd.name
                    );
                    return None;
                }

                let Some(create_time) = pod.metadata.creation_timestamp.clone() else {
                    tracing::warn!(
                        "Pod {namespace}:{pod_name} has no creation timestamp, skipping",
                        namespace = &pmd.namespace,
                        pod_name = &pmd.name
                    );
                    return None;
                };

                let lookback_start = Timestamp::now() - lookback_duration;

                tracing::info!(
                    "Pod {pod_name} create_time: {create_time} | lookback_start: {lookback_start}",
                    pod_name = &pmd.name,
                    create_time = create_time.0
                );

                let status = pod
                    .status
                    .as_ref()
                    .and_then(|s| s.phase.as_deref())
                    .unwrap_or("Unknown");
                tracing::info!(
                    "Pod [{:#?}] | CreateTime: {create_time} | Status: {status}",
                    &pmd,
                    create_time = create_time.0,
                    status = status
                );

                if create_time.0 >= lookback_start {
                    return None;
                }

                tracing::info!("Pod older than lookback start, so eligible for scaledown.");
                match find_root_object(kube_client.clone(), pod.clone().meta()).await {
                    Ok(obj) => Some(obj),
                    Err(e) => {
                        tracing::warn!("Failed to find root object! {e}");
                        tracing::info!(
                            "Skipping pod {namespace}:{pod_name} because it has no visible root object!",
                            namespace = &pmd.namespace,
                            pod_name = &pmd.name
                        );
                        None
                    }
                }
            }
        })
        .buffer_unordered(10)
        .collect()
        .await;

    let shutdown_events: HashSet<ScaleKind> = results.into_iter().flatten().collect();

    let num_shutdown_events = shutdown_events.len();

    futures::stream::iter(shutdown_events)
        .filter_map(|obj| async {
            if let Mode::DryRun = args.run_mode {
                tracing::info!(
                    "Dry-run: Would have sent [{}] {}:{} for scaledown",
                    obj.kind(),
                    obj.namespace().unwrap_or_default(),
                    obj.name()
                );
                None
            } else {
                Some(obj)
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

    Ok(QueryResponse {
        num_pods,
        shutdown_events: num_shutdown_events,
    })
}

#[cfg(test)]
mod tests {
    use minijinja::{Environment, context};
    use serde_json::json;

    const TEMPLATE: &str = include_str!("query.promql.j2");

    fn render(args: serde_json::Value) -> String {
        let env = Environment::new();
        env.render_str(TEMPLATE, context! { args }).unwrap()
    }

    #[test]
    fn query_uses_max_over_time() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            query.contains("max_over_time("),
            "should use max_over_time, not avg_over_time"
        );
        assert!(
            !query.contains("avg_over_time("),
            "should not contain avg_over_time"
        );
    }

    #[test]
    fn query_includes_gpu_util_fallback() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            query.contains("DCGM_FI_PROF_GR_ENGINE_ACTIVE"),
            "primary metric missing"
        );
        assert!(
            query.contains("DCGM_FI_DEV_GPU_UTIL"),
            "fallback metric missing"
        );
        assert!(
            query.contains("/ 100"),
            "fallback should normalize 0-100 to 0-1"
        );
    }

    #[test]
    fn query_without_power_threshold_has_no_unless() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            !query.contains("unless"),
            "should not have unless clause without power_threshold"
        );
        assert!(
            !query.contains("DCGM_FI_DEV_POWER_USAGE"),
            "should not reference power metric"
        );
    }

    #[test]
    fn query_with_power_threshold_adds_unless() {
        let query = render(json!({ "duration": 30, "power_threshold": 150.0 }));
        assert!(
            query.contains("unless on (exported_pod, exported_namespace)"),
            "should have unless clause"
        );
        assert!(
            query.contains("DCGM_FI_DEV_POWER_USAGE"),
            "should reference power metric"
        );
        assert!(
            query.contains(">= 150"),
            "should use the configured threshold"
        );
    }

    #[test]
    #[test]
    fn query_with_namespace_filter() {
        let query = render(json!({ "duration": 15, "namespace": "ml-team" }));
        let count = query.matches("exported_namespace =~ \"ml-team\"").count();
        // idle_gpus block appears twice (enriched + bare fallback), 2 metrics each = 4
        assert_eq!(
            count, 4,
            "namespace filter should appear in all compute metric selectors"
        );
    }

    #[test]
    fn query_with_namespace_and_power_threshold() {
        let query = render(json!({
            "duration": 15,
            "namespace": "ml-team",
            "power_threshold": 100.0
        }));
        let count = query.matches("exported_namespace =~ \"ml-team\"").count();
        // 4 from compute (2 paths x 2 metrics) + 1 from power = 5
        assert_eq!(
            count, 5,
            "namespace filter should appear in all metric selectors"
        );
    }

    #[test]
    fn query_with_model_name_filter() {
        let query = render(json!({ "duration": 30, "model_name": "NVIDIA A100" }));
        let count = query.matches("modelName =~ \"NVIDIA A100\"").count();
        // idle_gpus block appears twice (enriched + bare fallback), 2 metrics each = 4
        assert_eq!(
            count, 4,
            "model_name filter should appear in all compute metric selectors"
        );
    }

    #[test]
    fn query_duration_is_interpolated() {
        let query = render(json!({ "duration": 45 }));
        assert!(
            query.contains("[45m]"),
            "duration should be interpolated into range selector"
        );
    }

    #[test]
    fn query_default_uses_exported_labels() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            query.contains("exported_pod"),
            "default should use exported_pod"
        );
        assert!(
            query.contains("exported_namespace"),
            "default should use exported_namespace"
        );
        assert!(
            query.contains("exported_container"),
            "default should use exported_container"
        );
    }

    #[test]
    fn query_honor_labels_uses_native_labels() {
        let query = render(json!({ "duration": 30, "honor_labels": true }));
        assert!(
            !query.contains("exported_pod"),
            "honor_labels should not use exported_pod"
        );
        assert!(
            !query.contains("exported_namespace"),
            "honor_labels should not use exported_namespace"
        );
        assert!(
            query.contains("pod !="),
            "honor_labels should filter on pod"
        );
        assert!(
            query.contains("sum by (Hostname, container, pod, namespace"),
            "honor_labels should group by native labels"
        );
    }

    #[test]
    fn query_honor_labels_with_power_threshold() {
        let query = render(json!({
            "duration": 30,
            "honor_labels": true,
            "power_threshold": 120.0
        }));
        assert!(
            query.contains("unless on (pod, namespace)"),
            "honor_labels power unless should use native labels"
        );
    }
}
