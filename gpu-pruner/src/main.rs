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

use secrecy::ExposeSecret;

use gpu_pruner::{
    Meta, NamespaceMentionMapper, PendingScaleStatus, PodMetricData, QueryResponse,
    RootObjectError, ScaleKind, Scaler, TlsMode, ack_status, acknowledge_workload,
    check_pending_grace, fetch_workload, find_root_object, get_enabled_resources, get_prom_client,
    get_prometheus_token, get_slack_mentions, metrics::Metrics, set_pending_scale_at,
    slack::SlackNotifier,
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

    /// Specify enabled resources with a string of letters
    ///
    /// - `d` for Deployment
    /// - `r` for ReplicaSet
    /// - `s` for StatefulSet
    /// - `i` for InferenceService
    /// - `n` for Notebook
    /// - `l` for LeaderWorkerSet
    #[clap(short, long, default_value = "drsinl")]
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

    /// GPU utilization (0.0-1.0) below which a GPU counts as idle.
    /// DCGM GR_ENGINE_ACTIVE reports a small nonzero noise floor on otherwise
    /// idle GPUs, so a strict == 0 comparison misses them.
    #[clap(long, default_value = "0.01")]
    idle_threshold: f64,

    /// Regex of namespaces to exclude from pruning, applied as a negative
    /// match in the Prometheus query, eg. "infra-.*|monitoring"
    #[clap(long)]
    exclude_namespaces: Option<String>,

    /// Regex of pod names to exclude from pruning, applied as a negative
    /// match in the Prometheus query, eg. "dcgm-exporter-.*"
    #[clap(long)]
    exclude_pods: Option<String>,

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

    /// Address to serve Prometheus metrics on, eg. "0.0.0.0:8080".
    /// When unset, no metrics endpoint is exposed.
    #[clap(long)]
    #[serde(skip)]
    metrics_addr: Option<std::net::SocketAddr>,

    /// Slack incoming-webhook URL for idle notifications.
    /// Can also be set via the SLACK_WEBHOOK_URL env var.
    /// When unset, Slack notifications and the ack grace period are disabled.
    #[clap(long)]
    #[serde(skip)]
    slack_webhook_url: Option<String>,

    /// Slack channel override for notifications. When unset, messages go to
    /// the webhook's default channel.
    #[clap(long)]
    #[serde(skip)]
    slack_channel: Option<String>,

    /// Address to serve Slack interactive callbacks (ack button clicks) on,
    /// eg. "0.0.0.0:9090". Requires the SLACK_SIGNING_SECRET env var; requests
    /// failing Slack signature verification are rejected.
    #[clap(long)]
    #[serde(skip)]
    slack_interaction_addr: Option<std::net::SocketAddr>,

    /// Seconds to wait after a Slack notification before scaling down.
    /// Only applies when Slack notifications are enabled.
    #[clap(long, default_value = "300")]
    #[serde(skip)]
    ack_grace_period: u64,

    /// Namespace-to-Slack-mention mapping as JSON, eg.
    /// {"ml-team":"<@U123>","alice-":"<@UALICE>"}. Keys ending in "-" are
    /// namespace prefixes. Can also be set via SLACK_NAMESPACE_MENTIONS.
    #[clap(long)]
    #[serde(skip)]
    slack_namespace_mentions: Option<String>,
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

    let metrics = Metrics::new()?;

    let slack_ctx = build_slack_context(&args)?;

    let _slack_interaction_server = match args.slack_interaction_addr {
        Some(addr) => {
            let signing_secret = std::env::var("SLACK_SIGNING_SECRET").map_err(|_| {
                anyhow::anyhow!(
                    "--slack-interaction-addr requires the SLACK_SIGNING_SECRET env var; \
                     refusing to serve unauthenticated workload-mutating callbacks"
                )
            })?;
            let state = SlackInteractionState {
                kube_client: KubeClient::try_default().await?,
                signing_secret: std::sync::Arc::new(secrecy::SecretString::from(signing_secret)),
                http: reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()?,
                metrics: metrics.clone(),
            };
            Some(tokio::spawn(async move {
                if let Err(e) = serve_slack_interactions(addr, state).await {
                    tracing::error!("slack interaction server exited: {e}");
                }
            }))
        }
        None => None,
    };

    let _metrics_server = args.metrics_addr.map(|addr| {
        let metrics = metrics.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_metrics(addr, metrics).await {
                tracing::error!("metrics server exited: {e}");
            }
        })
    });

    let query_task = {
        let args = args.clone();
        let metrics = metrics.clone();
        let slack_ctx = slack_ctx.clone();
        tokio::spawn(async move {
            let mut interval =
                time::interval(tokio::time::Duration::from_secs(args.check_interval));
            loop {
                if args.daemon_mode {
                    interval.tick().await;
                }

                let client = build_prom_client(&args).await;
                match run_query_and_scale(
                    client,
                    query.clone(),
                    &args,
                    tx.clone(),
                    &metrics,
                    slack_ctx.as_ref(),
                )
                .await
                {
                    Ok(qr) => {
                        QUERY_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
                        metrics.query_successes.inc();
                        metrics.query_candidates.inc_by(qr.num_pods as u64);
                        metrics
                            .query_shutdown_events
                            .inc_by(qr.shutdown_events as u64);
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
                        metrics.query_failures.inc();
                        tracing::error!(
                            monotonic_counter.query_failures = 1,
                            "Failed to run query and scale down: {e}"
                        );
                        if failures > 5 {
                            tracing::error!("Too many consecutive failures, exiting");
                            break;
                        }
                    }
                }

                if !args.daemon_mode {
                    break;
                }
            }
            drop(tx);
        })
    };

    let scale_down_task = tokio::spawn({
        let metrics = metrics.clone();
        async move {
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
                    metrics.scale_failures.inc();
                    tracing::error!(
                        monotonic_counter.scale_failures = 1,
                        "Failed to scale resource! {e}"
                    );
                    continue;
                }

                let kind = sk.kind();
                let name = sk.name();
                let namespace = sk.namespace().unwrap_or_else(|| "default".to_string());

                metrics.scale_successes.inc();
                metrics.scales_by_kind.with_label_values(&[&kind]).inc();
                tracing::info!(
                    monotonic_counter.scale_successes = 1,
                    "Scaled Resource: [{kind}] - {namespace}:{name}",
                    kind = kind,
                    name = name,
                    namespace = namespace
                )
            }
        }
    });

    _ = tokio::try_join! {
        query_task,
        scale_down_task
    }?;

    Ok(())
}

#[derive(Clone)]
struct SlackContext {
    notifier: SlackNotifier,
    mentions: Option<NamespaceMentionMapper>,
    grace_secs: u64,
    stale_after_secs: u64,
}

fn build_slack_context(args: &Cli) -> anyhow::Result<Option<SlackContext>> {
    let webhook_url = args
        .slack_webhook_url
        .clone()
        .or_else(|| std::env::var("SLACK_WEBHOOK_URL").ok());

    let Some(webhook_url) = webhook_url else {
        tracing::info!("Slack notifications disabled (no webhook URL configured)");
        return Ok(None);
    };

    let notifier = SlackNotifier::new(webhook_url, args.slack_channel.clone())?;
    tracing::info!("Slack notifications enabled");

    let mentions = args
        .slack_namespace_mentions
        .clone()
        .or_else(|| std::env::var("SLACK_NAMESPACE_MENTIONS").ok())
        .map(|json_str| NamespaceMentionMapper::from_json(&json_str))
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid namespace mention mapping JSON: {e}"))?;

    if let Some(m) = &mentions {
        tracing::info!(
            "Namespace mention mapping enabled with {} entries",
            m.mappings.len()
        );
    }

    Ok(Some(SlackContext {
        notifier,
        mentions,
        grace_secs: args.ack_grace_period,
        stale_after_secs: args.ack_grace_period + 2 * args.check_interval,
    }))
}

#[derive(Clone)]
struct SlackInteractionState {
    kube_client: KubeClient,
    signing_secret: std::sync::Arc<secrecy::SecretString>,
    http: reqwest::Client,
    metrics: Metrics,
}

#[derive(Debug, serde::Deserialize)]
struct SlackInteractionPayload {
    user: SlackUser,
    actions: Vec<SlackAction>,
    response_url: String,
}

#[derive(Debug, serde::Deserialize)]
struct SlackUser {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    username: String,
}

#[derive(Debug, serde::Deserialize)]
struct SlackAction {
    value: String,
}

async fn serve_slack_interactions(
    addr: std::net::SocketAddr,
    state: SlackInteractionState,
) -> anyhow::Result<()> {
    use axum::{Router, routing::post};

    let app = Router::new()
        .route("/slack/interactions", post(handle_slack_interaction))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Serving Slack interaction callbacks on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_slack_interaction(
    axum::extract::State(state): axum::extract::State<SlackInteractionState>,
    headers: axum::http::HeaderMap,
    body: String,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let signature = headers
        .get("x-slack-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();

    if !gpu_pruner::slack::verify_slack_signature(
        state.signing_secret.expose_secret(),
        timestamp,
        &body,
        signature,
    ) {
        tracing::warn!("Rejected Slack interaction with invalid signature");
        return (StatusCode::UNAUTHORIZED, "invalid signature").into_response();
    }

    let payload_str = match serde_urlencoded::from_str::<Vec<(String, String)>>(&body) {
        Ok(params) => params
            .into_iter()
            .find(|(k, _)| k == "payload")
            .map(|(_, v)| v)
            .unwrap_or_default(),
        Err(e) => {
            tracing::error!("Failed to parse Slack form data: {e}");
            return (StatusCode::BAD_REQUEST, "invalid form data").into_response();
        }
    };

    if payload_str.is_empty() {
        return (StatusCode::OK, "OK").into_response();
    }

    let payload: SlackInteractionPayload = match serde_json::from_str(&payload_str) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to parse Slack payload JSON: {e}");
            return (StatusCode::BAD_REQUEST, "invalid payload").into_response();
        }
    };

    let Some(action) = payload.actions.first() else {
        return (StatusCode::BAD_REQUEST, "no action found").into_response();
    };

    // Button values are kind:namespace:name:hours; kinds and k8s names never
    // contain colons.
    let parts: Vec<&str> = action.value.split(':').collect();
    let [kind, namespace, name, duration_str] = parts.as_slice() else {
        tracing::error!("Invalid action value format: {}", action.value);
        return (StatusCode::BAD_REQUEST, "invalid action value").into_response();
    };

    let duration_hours: u32 = match duration_str.parse() {
        Ok(d @ 1..=168) => d,
        _ => {
            return (StatusCode::BAD_REQUEST, "invalid duration").into_response();
        }
    };

    let user = [&payload.user.name, &payload.user.username, &payload.user.id]
        .into_iter()
        .find(|s| !s.is_empty())
        .map(String::as_str)
        .unwrap_or("unknown");

    match acknowledge_workload(
        state.kube_client.clone(),
        kind,
        name,
        namespace,
        duration_hours,
        user,
    )
    .await
    {
        Ok(()) => {
            state.metrics.acknowledgments.inc();
            let response_message = serde_json::json!({
                "replace_original": true,
                "attachments": [{
                    "color": "good",
                    "title": "GPU Idle Acknowledgment Confirmed",
                    "text": format!(
                        "[{kind}] {namespace}:{name} will not be scaled down for the next {duration_hours} hours (acknowledged by {user})"
                    ),
                    "footer": "gpu-pruner",
                }]
            });
            if let Err(e) = state
                .http
                .post(&payload.response_url)
                .json(&response_message)
                .send()
                .await
            {
                tracing::error!("Failed to send response to Slack: {e}");
            }
            (StatusCode::OK, "Acknowledged").into_response()
        }
        Err(e) => {
            tracing::error!("Failed to acknowledge workload: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to acknowledge workload",
            )
                .into_response()
        }
    }
}

async fn serve_metrics(addr: std::net::SocketAddr, metrics: Metrics) -> anyhow::Result<()> {
    use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};

    async fn metrics_handler(State(metrics): State<Metrics>) -> axum::response::Response {
        match metrics.render() {
            Ok(body) => (
                [(
                    axum::http::header::CONTENT_TYPE,
                    "text/plain; version=0.0.4",
                )],
                body,
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to render metrics: {e}"),
            )
                .into_response(),
        }
    }

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(metrics);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Serving Prometheus metrics on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn build_prom_client(args: &Cli) -> Client {
    let token = get_prometheus_token()
        .await
        .expect("failed to get prometheus token");
    get_prom_client(
        &args.prometheus_url,
        token,
        args.prometheus_tls_mode,
        args.prometheus_tls_cert.clone(),
    )
    .expect("failed to build prometheus client")
}

#[tracing::instrument(skip_all)]
async fn run_query_and_scale(
    client: Client,
    query: String,
    args: &Cli,
    tx: Sender<ScaleKind>,
    metrics: &Metrics,
    slack: Option<&SlackContext>,
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

    // Dedup by (pod, namespace) before processing. Multi-GPU pods produce one
    // series per GPU, but we only need to resolve the owner chain once per pod.
    let num_pods = vec.len();
    let mut seen_pods = HashSet::new();
    let unique_pods: Vec<_> = vec
        .into_iter()
        .filter_map(|v| {
            let pmd: PodMetricData = match (&v).try_into() {
                Ok(pmd) => pmd,
                Err(e) => {
                    tracing::error!("Failed to unwrap pod fields: {e}");
                    return None;
                }
            };
            let key = (pmd.name.clone(), pmd.namespace.clone());
            if seen_pods.insert(key) {
                Some(pmd)
            } else {
                None
            }
        })
        .collect();

    tracing::info!(
        "Query returned {num_pods} series across {} unique pods",
        unique_pods.len()
    );
    metrics.pods_checked.set(unique_pods.len() as i64);

    // Process pods concurrently (up to 10 at a time) instead of serially.
    // Each pod requires 1-3 API calls (get pod, walk owner refs), so parallelism
    // cuts wall-clock time significantly on large result sets.
    let results: Vec<Option<ScaleKind>> = futures::stream::iter(unique_pods)
        .map(|pmd| {
            let kube_client = kube_client.clone();
            async move {

                let api = Api::<Pod>::namespaced(kube_client.clone(), &pmd.namespace);
                let pod = match api.get_opt(&pmd.name).await {
                    Ok(Some(pod)) => pod,
                    Ok(None) => {
                        tracing::info!(
                            "Skipping {ns}:{name}, pod no longer exists",
                            ns = &pmd.namespace,
                            name = &pmd.name
                        );
                        return None;
                    }
                    Err(e) => {
                        tracing::error!(
                            "Skipping {ns}:{name}, retrieval error: {e}",
                            ns = &pmd.namespace,
                            name = &pmd.name
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
                let status = pod
                    .status
                    .as_ref()
                    .and_then(|s| s.phase.as_deref())
                    .unwrap_or("Unknown");

                tracing::info!(
                    "Pod {ns}:{name} | status={status} | created={created} | lookback={lookback_start}",
                    ns = &pmd.namespace,
                    name = &pmd.name,
                    created = create_time.0,
                );

                if create_time.0 >= lookback_start {
                    return None;
                }

                tracing::info!(
                    "Pod {ns}:{name} is idle and eligible for scaledown",
                    ns = &pmd.namespace,
                    name = &pmd.name,
                );
                match find_root_object(kube_client.clone(), pod.meta()).await {
                    Ok(obj) => Some(obj),
                    Err(e @ RootObjectError::NonScalable { .. }) => {
                        tracing::debug!(
                            "Skipping {ns}:{name}, {e}",
                            ns = &pmd.namespace,
                            name = &pmd.name,
                        );
                        None
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Skipping {ns}:{name}, no scalable root object: {e}",
                            ns = &pmd.namespace,
                            name = &pmd.name,
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
    metrics.idle_workloads.set(num_shutdown_events as i64);

    match slack {
        Some(slack) if slack.grace_secs > 0 => {
            dispatch_with_ack_grace(kube_client, shutdown_events, args, tx, metrics, slack).await;
        }
        _ => {
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
        }
    }

    Ok(QueryResponse {
        num_pods,
        shutdown_events: num_shutdown_events,
    })
}

/// Scale-down dispatch with Slack notify-then-wait semantics: notify and
/// annotate on first detection, skip while the grace period runs, scale once
/// it expires unless an acknowledgment landed in the meantime.
async fn dispatch_with_ack_grace(
    kube_client: KubeClient,
    shutdown_events: HashSet<ScaleKind>,
    args: &Cli,
    tx: Sender<ScaleKind>,
    metrics: &Metrics,
    slack: &SlackContext,
) {
    let mut acknowledged_count: i64 = 0;

    for obj in shutdown_events {
        let ack = ack_status(&obj);
        let kind = obj.kind();
        let name = obj.name();
        let namespace = obj.namespace().unwrap_or_else(|| "default".to_string());

        if ack.acknowledged {
            acknowledged_count += 1;
            metrics.scaledowns_prevented.inc();
            tracing::info!(
                "Skipping [{kind}] {namespace}:{name} - acknowledged until {} by {}",
                ack.expires_at.as_deref().unwrap_or("unknown"),
                ack.by_user.as_deref().unwrap_or("unknown"),
            );
            continue;
        }

        match check_pending_grace(&obj, slack.grace_secs, slack.stale_after_secs) {
            PendingScaleStatus::NotPending | PendingScaleStatus::Stale => {
                if matches!(args.run_mode, Mode::DryRun) {
                    tracing::info!(
                        "Dry-run: Would notify [{kind}] {namespace}:{name} and wait {}s before scale-down",
                        slack.grace_secs
                    );
                    continue;
                }

                let mentions = get_slack_mentions(&obj, slack.mentions.as_ref());
                match slack
                    .notifier
                    .send_notification(&obj, args.duration, slack.grace_secs, mentions)
                    .await
                {
                    Ok(()) => metrics.slack_notifications_sent.inc(),
                    Err(e) => {
                        metrics.slack_notification_failures.inc();
                        tracing::error!("Failed to send Slack notification: {e}");
                    }
                }

                if let Err(e) =
                    set_pending_scale_at(kube_client.clone(), &kind, &name, &namespace).await
                {
                    tracing::error!(
                        "Failed to set pending-scale annotation for [{kind}] {namespace}:{name}: {e}"
                    );
                } else {
                    tracing::info!(
                        "Started {}s ack grace period for [{kind}] {namespace}:{name}",
                        slack.grace_secs
                    );
                }
            }
            PendingScaleStatus::InGrace { until } => {
                tracing::info!(
                    "Skipping [{kind}] {namespace}:{name} - ack grace period until {}",
                    until.to_rfc3339()
                );
            }
            PendingScaleStatus::GraceExpired => {
                // Re-fetch so an ack applied after this scan's listing is seen.
                let fresh = match fetch_workload(kube_client.clone(), &kind, &name, &namespace)
                    .await
                {
                    Ok(fresh) => fresh,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to re-fetch [{kind}] {namespace}:{name}, using cached object: {e}"
                        );
                        obj.clone()
                    }
                };

                if ack_status(&fresh).acknowledged {
                    acknowledged_count += 1;
                    metrics.scaledowns_prevented.inc();
                    tracing::info!(
                        "Skipping [{kind}] {namespace}:{name} - acknowledged during grace period"
                    );
                    continue;
                }

                if matches!(args.run_mode, Mode::DryRun) {
                    tracing::info!(
                        "Dry-run: Would scale [{kind}] {namespace}:{name} after grace period expired"
                    );
                    continue;
                }

                tracing::info!(
                    "Sending [{kind}] {namespace}:{name} for scaledown after grace period expired"
                );
                if let Err(e) = tx.send(fresh).await {
                    tracing::error!("Failed to send object for scaledown: {:?}", e);
                }
            }
        }
    }

    metrics.acknowledged_workloads.set(acknowledged_count);
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
    fn query_uses_idle_threshold_not_strict_zero() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            query.contains("< 0.01"),
            "default idle threshold should be 0.01, not == 0"
        );
        assert!(!query.contains("== 0"), "should not use strict == 0");
    }

    #[test]
    fn query_idle_threshold_is_configurable() {
        let query = render(json!({ "duration": 30, "idle_threshold": 0.05 }));
        assert!(
            query.contains("< 0.05"),
            "should use configured idle threshold"
        );
    }

    #[test]
    fn query_without_excludes_has_no_negative_matchers() {
        let query = render(json!({ "duration": 30 }));
        assert!(
            !query.contains("!~"),
            "no negative matchers unless exclude flags are set"
        );
    }

    #[test]
    fn query_exclude_flags_add_negative_matchers() {
        let query = render(json!({
            "duration": 30,
            "exclude_namespaces": "infra-.*|monitoring",
            "exclude_pods": "dcgm-exporter-.*",
        }));
        // idle_gpus block appears twice (enriched + bare fallback), 2 metrics each = 4
        assert_eq!(
            query
                .matches("exported_namespace !~ \"infra-.*|monitoring\"")
                .count(),
            4,
            "namespace exclude should appear in all compute metric selectors"
        );
        assert_eq!(
            query
                .matches("exported_pod !~ \"dcgm-exporter-.*\"")
                .count(),
            4,
            "pod exclude should appear in all compute metric selectors"
        );
    }

    #[test]
    fn query_exclude_flags_apply_to_power_selector() {
        let query = render(json!({
            "duration": 30,
            "power_threshold": 150.0,
            "exclude_namespaces": "infra-.*",
            "exclude_pods": "dcgm-exporter-.*",
        }));
        assert_eq!(
            query.matches("exported_namespace !~ \"infra-.*\"").count(),
            5,
            "namespace exclude should also appear in the power metric selector"
        );
        assert_eq!(
            query
                .matches("exported_pod !~ \"dcgm-exporter-.*\"")
                .count(),
            5,
            "pod exclude should also appear in the power metric selector"
        );
    }

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
