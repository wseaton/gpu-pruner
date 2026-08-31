export type TimeWindow = "1h" | "7d" | "30d";

export interface ScaleDownsSummary {
  lifetime: number;
}

export interface IdleWorkloadsSummary {
  current: number;
}

export interface SummaryResponse {
  scale_downs: ScaleDownsSummary;
  idle_workloads: IdleWorkloadsSummary;
  pods_checked: number;
  updated_at: string;
}

export interface ScaleDownsStats {
  lifetime: number;
  in_window?: number;
  window: string;
}

export interface StatsResponse {
  scale_downs: ScaleDownsStats;
  idle_workloads: IdleWorkloadsSummary;
  prometheus_available: boolean;
  pods_checked: number;
  updated_at: string;
}

export interface ClusterInfo {
  name: string;
  honor_labels: boolean;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchSummary(): Promise<SummaryResponse> {
  return parseJson<SummaryResponse>(await fetch("/api/v1/summary"));
}

export async function fetchStats(window: TimeWindow): Promise<StatsResponse> {
  return parseJson<StatsResponse>(
    await fetch(`/api/v1/stats?window=${encodeURIComponent(window)}`),
  );
}

export async function fetchClusters(): Promise<ClusterInfo[]> {
  return parseJson<ClusterInfo[]>(await fetch("/api/v1/clusters"));
}

export interface IdleGpuHoursEntry {
  rank: number;
  cluster: string;
  namespace: string;
  pod: string;
  idle_hours: number;
}

export interface IdleGpuHoursResponse {
  entries: IdleGpuHoursEntry[];
  prometheus_available: boolean;
  updated_at: string;
}

const IDLE_GPU_HOURS_QUERY_EXPORTED =
  'sort_desc(sum by (exported_namespace, exported_pod) (count_over_time((DCGM_FI_PROF_GR_ENGINE_ACTIVE{exported_pod!="",exported_pod!~"dcgm-exporter-.*"} < 0.01)[7d:1m]) / 60))';

const IDLE_GPU_HOURS_QUERY_HONOR =
  'sort_desc(sum by (namespace, pod) (count_over_time((DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod!="",pod!~"dcgm-exporter-.*"} < 0.01)[7d:1m]) / 60))';

const IDLE_GPU_HOURS_LIMIT = 25;

interface PrometheusInstantVectorResponse {
  status?: string;
  data?: {
    resultType?: string;
    result?: Array<{
      metric?: Record<string, string>;
      value?: [number, string];
    }>;
  };
}

function parseIdleGpuEntries(
  body: PrometheusInstantVectorResponse,
  cluster: string,
): IdleGpuHoursEntry[] {
  if (body.status !== "success") {
    return [];
  }

  const results = body.data?.result ?? [];
  const entries: IdleGpuHoursEntry[] = [];

  for (const result of results) {
    if (entries.length >= IDLE_GPU_HOURS_LIMIT) {
      break;
    }

    const namespace =
      result.metric?.namespace ?? result.metric?.exported_namespace;
    const pod = result.metric?.pod ?? result.metric?.exported_pod;
    if (!namespace || !pod) {
      continue;
    }

    const raw = result.value?.[1];
    if (raw == null) {
      continue;
    }

    const idleHours = Number(raw);
    if (!Number.isFinite(idleHours)) {
      continue;
    }

    entries.push({
      rank: 0,
      cluster,
      namespace,
      pod,
      idle_hours: idleHours,
    });
  }

  return entries;
}

export async function fetchIdleGpuHours(
  cluster: string,
  honorLabels: boolean,
): Promise<IdleGpuHoursResponse> {
  const updatedAt = new Date().toISOString();
  const query = honorLabels
    ? IDLE_GPU_HOURS_QUERY_HONOR
    : IDLE_GPU_HOURS_QUERY_EXPORTED;

  try {
    const response = await fetch(
      `/prom/${encodeURIComponent(cluster)}/api/v1/query?query=${encodeURIComponent(query)}`,
    );

    if (!response.ok) {
      return { entries: [], prometheus_available: false, updated_at: updatedAt };
    }

    const body = (await response.json()) as PrometheusInstantVectorResponse;
    const entries = parseIdleGpuEntries(body, cluster);
    entries.forEach((e, i) => { e.rank = i + 1; });

    return {
      entries,
      prometheus_available: body.status === "success",
      updated_at: updatedAt,
    };
  } catch {
    return { entries: [], prometheus_available: false, updated_at: updatedAt };
  }
}

export async function fetchAllClustersIdleGpuHours(
  clusters: ClusterInfo[],
): Promise<IdleGpuHoursResponse> {
  const updatedAt = new Date().toISOString();
  const results = await Promise.all(
    clusters.map((c) => fetchIdleGpuHours(c.name, c.honor_labels)),
  );

  const allEntries = results.flatMap((r) => r.entries);
  allEntries.sort((a, b) => b.idle_hours - a.idle_hours);
  allEntries.forEach((e, i) => { e.rank = i + 1; });

  return {
    entries: allEntries.slice(0, IDLE_GPU_HOURS_LIMIT),
    prometheus_available: results.some((r) => r.prometheus_available),
    updated_at: updatedAt,
  };
}
