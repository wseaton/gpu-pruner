# gpu-pruner

The `gpu-pruner` is a non-destructive idle culler that works with Red Hat OpenShift AI/Kubeflow provided APIs (`InferenceService` and `Notebook`), as well as generic `Deployment`, `ReplicaSet`, `StatefulSet` and `LeaderWorkerSet` (`leaderworkerset.x-k8s.io`).

The way it works is by querying cluster NVIDIA DCGM metrics and looking at a window of GPU utilization per pod. A scaling decision is made by looking up the pods metadata, and using owner-references to figure out the owning resource.

**Note:** Requires a K8s service account with CRUD access to the resources in the namespaces that you want to prune.

An [example set of k8s deployment manifests](./gpu-pruner/hack/kustomization.yaml) are available along with the role bindings to run in "cluster mode". 

[prebuilt images](https://github.com/wseaton/gpu-pruner/pkgs/container/gpu-pruner) based on the Dockerfiles in the repository are published to ghcr.io.

## background

The background for `gpu-pruner` is that in certain environments it is very easy for cluster users to request GPUs and then (either accidentally or not accidentally) not consume GPU resources. We needed a method to proactively identify this type of use, and scale down workloads that are idle from the GPU hardware perspective, compared to the default for `Notebook` resources which is web activity. It is totally possible for a user to consume a GPU from a pod PoV but never actually run a workload on it!

This culler politely pauses workloads that appear idle by scaling them down to 0 replicas. Features may be added in the future for better notifications, but the idea is that a user can simply re-enable the workload when they are ready to test/demo again.

## usage 

```sh
Usage: gpu-pruner [OPTIONS] --prometheus-url <PROMETHEUS_URL>

Options:
  -t, --duration <DURATION>
          time in minutes of no gpu activity to use for pruning

          [default: 30]

  -d, --daemon-mode
          daemon mode to run in, if true, will run indefinitely

  -e, --enabled-resources <ENABLED_RESOURCES>
          Specifcy enabled resources with a string of letters

          - `d` for Deployment - `r` for ReplicaSet - `s` for StatefulSet - `i` for InferenceService - `n` for Notebook - `l` for LeaderWorkerSet

          [default: drsinl]

  -c, --check-interval <CHECK_INTERVAL>
          interval in seconds to check for idle pods, only used in daemon mode

          [default: 180]

  -n, --namespace <NAMESPACE>
          namespace to use for search filter, is passed down to prometheus as a pattern match

  -g, --grace-period <GRACE_PERIOD>
          Seconds of grace period to allow for metrics to be published

          [default: 300]

  -m, --model-name <MODEL_NAME>
          model name of GPU to use for filter, eg. "NVIDIA A10G", is passed down to prometheus as a pattern match

      --power-threshold <POWER_THRESHOLD>
          Power draw threshold in watts. When set, GPUs showing peak power usage above this value over the lookback window are excluded from idle candidates even if compute utilization is zero. Useful as a corroborating signal (e.g. 100 for A10G, 150 for A100/H100)

      --idle-threshold <IDLE_THRESHOLD>
          GPU utilization (0.0-1.0) below which a GPU counts as idle. DCGM GR_ENGINE_ACTIVE reports a small nonzero noise floor on otherwise idle GPUs, so a strict == 0 comparison misses them

          [default: 0.01]

      --exclude-namespaces <EXCLUDE_NAMESPACES>
          Regex of namespaces to exclude from pruning, applied as a negative match in the Prometheus query, eg. "infra-.*|monitoring"

      --exclude-pods <EXCLUDE_PODS>
          Regex of pod names to exclude from pruning, applied as a negative match in the Prometheus query, eg. "dcgm-exporter-.*"

  -r, --run-mode <RUN_MODE>
          Operation mode of the scaler process

          [default: dry-run]
          [possible values: scale-down, dry-run]

      --prometheus-url <PROMETHEUS_URL>
          Prometheus URL to query for GPU metrics eg. "http://prometheus-k8s.openshift-monitoring.svc:9090"

      --prometheus-token <PROMETHEUS_TOKEN>
          Prometheus token to use for authentication, if not provided, will try to authenticate using the service token of the currently logged in K8s user

  -l, --log-format <LOG_FORMAT>
          Log format to use

          [default: default]
          [possible values: json, default, pretty]

  -h, --help
          Print help (see a summary with '-h')
```


## OTEL via OTLP

When compiled with the `otel` feature, OTLP metrics and trace export is enabled, and can be configured via environment variables, eg:

```
          env:
            - name: NODE_IP
              valueFrom:
                fieldRef:
                  apiVersion: v1
                  fieldPath: status.hostIP
            - name: OTEL_TRACES_EXPORTER
              value: otlp
            - name: OTEL_METRICS_EXPORTER
              value: otlp
            - name: OTEL_EXPORTER_OTLP_METRICS_ENDPOINT
              value: 'http://$(NODE_IP):4317'
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: 'http://$(NODE_IP):4317'
```

