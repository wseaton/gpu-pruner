use std::sync::Arc;

use prometheus::{Encoder, IntCounter, IntCounterVec, IntGauge, Opts, Registry, TextEncoder};

/// Prometheus metrics for the pruner, backed by a private registry.
///
/// Cheap to clone; all handles share the same underlying registry.
#[derive(Clone)]
pub struct Metrics {
    registry: Arc<Registry>,

    pub query_successes: IntCounter,
    pub query_failures: IntCounter,
    pub query_candidates: IntCounter,
    pub query_shutdown_events: IntCounter,

    pub scale_successes: IntCounter,
    pub scale_failures: IntCounter,
    pub scales_by_kind: IntCounterVec,

    pub idle_workloads: IntGauge,
    pub pods_checked: IntGauge,
}

impl Metrics {
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        let query_successes = IntCounter::new(
            "gpu_pruner_query_successes_total",
            "Total number of successful Prometheus queries",
        )?;
        let query_failures = IntCounter::new(
            "gpu_pruner_query_failures_total",
            "Total number of failed Prometheus queries",
        )?;
        let query_candidates = IntCounter::new(
            "gpu_pruner_query_candidates_total",
            "Total number of idle GPU candidates found across all queries",
        )?;
        let query_shutdown_events = IntCounter::new(
            "gpu_pruner_query_shutdown_events_total",
            "Total number of shutdown events detected",
        )?;
        let scale_successes = IntCounter::new(
            "gpu_pruner_scale_successes_total",
            "Total number of successful scale operations",
        )?;
        let scale_failures = IntCounter::new(
            "gpu_pruner_scale_failures_total",
            "Total number of failed scale operations",
        )?;
        let scales_by_kind = IntCounterVec::new(
            Opts::new(
                "gpu_pruner_scales_by_kind_total",
                "Total scale-down operations by resource kind",
            ),
            &["kind"],
        )?;
        let idle_workloads = IntGauge::new(
            "gpu_pruner_idle_workloads",
            "Number of workloads eligible for scale-down in the last check",
        )?;
        let pods_checked = IntGauge::new(
            "gpu_pruner_pods_checked",
            "Number of pods analyzed in the last query",
        )?;

        registry.register(Box::new(query_successes.clone()))?;
        registry.register(Box::new(query_failures.clone()))?;
        registry.register(Box::new(query_candidates.clone()))?;
        registry.register(Box::new(query_shutdown_events.clone()))?;
        registry.register(Box::new(scale_successes.clone()))?;
        registry.register(Box::new(scale_failures.clone()))?;
        registry.register(Box::new(scales_by_kind.clone()))?;
        registry.register(Box::new(idle_workloads.clone()))?;
        registry.register(Box::new(pods_checked.clone()))?;

        Ok(Self {
            registry: Arc::new(registry),
            query_successes,
            query_failures,
            query_candidates,
            query_shutdown_events,
            scale_successes,
            scale_failures,
            scales_by_kind,
            idle_workloads,
            pods_checked,
        })
    }

    pub fn render(&self) -> anyhow::Result<String> {
        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&self.registry.gather(), &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics::Metrics;

    #[test]
    fn metrics_register_and_render() {
        let m = Metrics::new().unwrap();
        m.scale_successes.inc();
        m.idle_workloads.set(3);
        m.pods_checked.set(10);
        m.scales_by_kind.with_label_values(&["Deployment"]).inc();

        let rendered = m.render().unwrap();
        assert!(rendered.contains("gpu_pruner_scale_successes_total 1"));
        assert!(rendered.contains("gpu_pruner_idle_workloads 3"));
        assert!(rendered.contains("gpu_pruner_pods_checked 10"));
        assert!(rendered.contains("gpu_pruner_scales_by_kind_total{kind=\"Deployment\"} 1"));
    }

    #[test]
    fn cloned_metrics_share_registry() {
        let m = Metrics::new().unwrap();
        let m2 = m.clone();
        m2.query_successes.inc();
        assert!(
            m.render()
                .unwrap()
                .contains("gpu_pruner_query_successes_total 1")
        );
    }
}
