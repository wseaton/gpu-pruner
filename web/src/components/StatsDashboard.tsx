import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Card,
  CardBody,
  CardTitle,
  FormGroup,
  Grid,
  GridItem,
  MenuToggle,
  Select,
  SelectList,
  SelectOption,
  Skeleton,
  Title,
} from "@patternfly/react-core";
import { fetchStats, type StatsResponse, type TimeWindow } from "../api";

const REFRESH_INTERVAL_MS = 30_000;

const WINDOW_LABELS: Record<TimeWindow, string> = {
  "1h": "Last hour",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
};

function formatUpdatedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString();
}

interface StatCardProps {
  title: string;
  value: number | null;
  subtitle: string;
  accent: "danger" | "warning";
  loading: boolean;
}

function StatCard({ title, value, subtitle, accent, loading }: StatCardProps) {
  const borderColor = accent === "danger" ? "#c9190b" : "#f0ab00";

  return (
    <Card style={{ borderTop: `4px solid ${borderColor}`, height: "100%" }}>
      <CardTitle>{title}</CardTitle>
      <CardBody>
        {loading ? (
          <Skeleton height="48px" width="80px" />
        ) : (
          <Title headingLevel="h1" size="4xl">
            {value ?? "—"}
          </Title>
        )}
        <p style={{ marginTop: "0.5rem", color: "#6a6e73" }}>{subtitle}</p>
      </CardBody>
    </Card>
  );
}

export function StatsDashboard() {
  const [timeWindow, setTimeWindow] = useState<TimeWindow>("7d");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isWindowOpen, setIsWindowOpen] = useState(false);

  const loadStats = useCallback(async () => {
    try {
      setError(null);
      const data = await fetchStats(timeWindow);
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load stats");
    } finally {
      setLoading(false);
    }
  }, [timeWindow]);

  useEffect(() => {
    setLoading(true);
    void loadStats();
  }, [loadStats]);

  useEffect(() => {
    const timer = globalThis.setInterval(() => {
      void loadStats();
    }, REFRESH_INTERVAL_MS);
    return () => globalThis.clearInterval(timer);
  }, [loadStats]);

  const scaleDownValue =
    stats?.prometheus_available && stats.scale_downs.in_window != null
      ? stats.scale_downs.in_window
      : stats?.scale_downs.lifetime ?? null;

  const scaleDownSubtitle = stats?.prometheus_available
    ? `${WINDOW_LABELS[timeWindow]} (Prometheus)`
    : "Since pod start (Prometheus unavailable)";

  return (
    <>
      {error && (
        <Alert variant="danger" title="Failed to load dashboard data" isInline>
          {error}
        </Alert>
      )}

      {!stats?.prometheus_available && stats && !loading && (
        <Alert
          variant="warning"
          title="Prometheus range unavailable"
          isInline
          style={{ marginTop: "1rem" }}
        >
          Showing scale-down total since this pod started. Range counts require
          Prometheus to scrape gpu-pruner metrics.
        </Alert>
      )}

      <FormGroup label="Scale-down time range" style={{ marginTop: "1.5rem" }}>
        <Select
          selected={timeWindow}
          isOpen={isWindowOpen}
          onOpenChange={(open) => setIsWindowOpen(open)}
          onSelect={(_event, value) => {
            if (value) {
              setTimeWindow(value as TimeWindow);
            }
            setIsWindowOpen(false);
          }}
          toggle={(toggleRef) => (
            <MenuToggle
              ref={toggleRef}
              onClick={() => setIsWindowOpen((open) => !open)}
              isExpanded={isWindowOpen}
            >
              {WINDOW_LABELS[timeWindow]}
            </MenuToggle>
          )}
        >
          <SelectList>
            {(Object.keys(WINDOW_LABELS) as TimeWindow[]).map((key) => (
              <SelectOption key={key} value={key}>
                {WINDOW_LABELS[key]}
              </SelectOption>
            ))}
          </SelectList>
        </Select>
      </FormGroup>

      <Grid hasGutter style={{ marginTop: "1.5rem" }}>
        <GridItem sm={12} md={6}>
          <StatCard
            title="Scale-downs"
            value={scaleDownValue}
            subtitle={scaleDownSubtitle}
            accent="danger"
            loading={loading}
          />
        </GridItem>
        <GridItem sm={12} md={6}>
          <StatCard
            title="Idle workloads"
            value={stats?.idle_workloads.current ?? null}
            subtitle="Eligible for scale-down in last check"
            accent="warning"
            loading={loading}
          />
        </GridItem>
      </Grid>

      {stats && !loading && (
        <p style={{ marginTop: "1.5rem", color: "#6a6e73", fontSize: "0.875rem" }}>
          Pods checked in last cycle: {stats.pods_checked}. Updated{" "}
          {formatUpdatedAt(stats.updated_at)}. Auto-refreshes every 30 seconds.
        </p>
      )}
    </>
  );
}
