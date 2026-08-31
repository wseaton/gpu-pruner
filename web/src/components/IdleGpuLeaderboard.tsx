import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Card,
  CardBody,
  CardTitle,
  FormGroup,
  MenuToggle,
  Select,
  SelectList,
  SelectOption,
  Skeleton,
} from "@patternfly/react-core";
import {
  fetchAllClustersIdleGpuHours,
  fetchClusters,
  fetchIdleGpuHours,
  type ClusterInfo,
  type IdleGpuHoursResponse,
} from "../api";

const REFRESH_INTERVAL_MS = 30_000;
const ALL_CLUSTERS = "__all__";

function formatIdleHours(value: number): string {
  return value.toFixed(1);
}

function formatUpdatedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString();
}

export function IdleGpuLeaderboard() {
  const [clusters, setClusters] = useState<ClusterInfo[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<string>(ALL_CLUSTERS);
  const [isClusterOpen, setIsClusterOpen] = useState(false);
  const [data, setData] = useState<IdleGpuHoursResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchClusters()
      .then((c) => {
        setClusters(c);
        if (c.length === 1) {
          setSelectedCluster(c[0].name);
        }
      })
      .catch(() => {});
  }, []);

  const load = useCallback(async () => {
    if (clusters.length === 0) return;

    try {
      setError(null);
      let response: IdleGpuHoursResponse;
      if (selectedCluster === ALL_CLUSTERS) {
        response = await fetchAllClustersIdleGpuHours(clusters);
      } else {
        const info = clusters.find((c) => c.name === selectedCluster);
        response = await fetchIdleGpuHours(
          selectedCluster,
          info?.honor_labels ?? false,
        );
      }
      setData(response);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load idle GPU hours",
      );
    } finally {
      setLoading(false);
    }
  }, [clusters, selectedCluster]);

  useEffect(() => {
    setLoading(true);
    void load();
  }, [load]);

  useEffect(() => {
    const timer = globalThis.setInterval(() => {
      void load();
    }, REFRESH_INTERVAL_MS);
    return () => globalThis.clearInterval(timer);
  }, [load]);

  const showClusterColumn =
    selectedCluster === ALL_CLUSTERS && clusters.length > 1;

  const clusterLabel =
    selectedCluster === ALL_CLUSTERS
      ? "All Clusters"
      : selectedCluster;

  return (
    <Card style={{ marginTop: "1.5rem" }}>
      <CardTitle>Top Idle GPU Hours (7 Days)</CardTitle>
      <CardBody>
        {error && (
          <Alert variant="danger" title="Failed to load leaderboard" isInline>
            {error}
          </Alert>
        )}

        {!data?.prometheus_available && data && !loading && !error && (
          <Alert
            variant="warning"
            title="Prometheus unavailable"
            isInline
            style={{ marginBottom: "1rem" }}
          >
            Could not query Prometheus for idle GPU hours. Check that the
            dashboard can reach the configured Prometheus URL(s).
          </Alert>
        )}

        {clusters.length > 1 && (
          <FormGroup label="Cluster" style={{ marginBottom: "1rem" }}>
            <Select
              selected={selectedCluster}
              isOpen={isClusterOpen}
              onOpenChange={(open) => setIsClusterOpen(open)}
              onSelect={(_event, value) => {
                if (value) {
                  setSelectedCluster(value as string);
                }
                setIsClusterOpen(false);
              }}
              toggle={(toggleRef) => (
                <MenuToggle
                  ref={toggleRef}
                  onClick={() => setIsClusterOpen((open) => !open)}
                  isExpanded={isClusterOpen}
                >
                  {clusterLabel}
                </MenuToggle>
              )}
            >
              <SelectList>
                <SelectOption key={ALL_CLUSTERS} value={ALL_CLUSTERS}>
                  All Clusters
                </SelectOption>
                {clusters.map((c) => (
                  <SelectOption key={c.name} value={c.name}>
                    {c.name}
                  </SelectOption>
                ))}
              </SelectList>
            </Select>
          </FormGroup>
        )}

        {loading ? (
          <Skeleton height="200px" />
        ) : data?.prometheus_available && data.entries.length === 0 ? (
          <p style={{ color: "#6a6e73" }}>No idle GPU hours found.</p>
        ) : data?.prometheus_available ? (
          <>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "0.875rem",
              }}
            >
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "2px solid #d2d2d2" }}>
                  <th style={{ padding: "0.5rem 0.75rem" }}>Rank</th>
                  {showClusterColumn && (
                    <th style={{ padding: "0.5rem 0.75rem" }}>Cluster</th>
                  )}
                  <th style={{ padding: "0.5rem 0.75rem" }}>Namespace</th>
                  <th style={{ padding: "0.5rem 0.75rem" }}>Pod</th>
                  <th style={{ padding: "0.5rem 0.75rem", textAlign: "right" }}>
                    Idle Hours
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.entries.map((entry) => (
                  <tr
                    key={`${entry.cluster}/${entry.namespace}/${entry.pod}`}
                    style={{ borderBottom: "1px solid #f0f0f0" }}
                  >
                    <td style={{ padding: "0.5rem 0.75rem" }}>{entry.rank}</td>
                    {showClusterColumn && (
                      <td style={{ padding: "0.5rem 0.75rem" }}>
                        {entry.cluster}
                      </td>
                    )}
                    <td style={{ padding: "0.5rem 0.75rem" }}>
                      {entry.namespace}
                    </td>
                    <td style={{ padding: "0.5rem 0.75rem" }}>{entry.pod}</td>
                    <td
                      style={{
                        padding: "0.5rem 0.75rem",
                        textAlign: "right",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {formatIdleHours(entry.idle_hours)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p
              style={{
                marginTop: "1rem",
                color: "#6a6e73",
                fontSize: "0.875rem",
              }}
            >
              Updated {formatUpdatedAt(data.updated_at)}. Auto-refreshes every
              30 seconds.
            </p>
          </>
        ) : null}
      </CardBody>
    </Card>
  );
}
