# gpu-pruner

The `gpu-pruner` is a non-desctuctive idle culler that works with RHOAI/Kubeflow provided APIs (`InferenceService` and `Notebook`), as well as generic `Deployment`, `ReplicaSet` and `StatefulSet`.

The way it works is by querying cluster NVIDIA DCGM metrics and looking at a window of GPU utiliazation per pod. A scaling decision is made by looking up the pods metadata, and using owner-references to figure out the owning resource.

**Note:** Requires a K8s service account with CRUD access to the resources in the namespaces that you want to prune.

## usage 

```sh
Usage: gpu-pruner [OPTIONS] --prometheus-url <PROMETHEUS_URL>

Options:
  -t, --duration <DURATION>
          time in minutes of no gpu activity to use for pruning

          [default: 30]

  -d, --daemon-mode
          whether or not to run in daemon mode

  -c, --check-interval <CHECK_INTERVAL>
          interval in seconds to check for idle pods, only used in daemon mode

          [default: 180]

  -n, --namespace <NAMESPACE>
          namespace to use for search filter, is passed down to prometheus as a pattern match

  -m, --model-name <MODEL_NAME>
          model name of GPU to use for filter, eg. "NVIDIA A10G", is passed down to prometheus as a pattern match

  -r, --run-mode <RUN_MODE>
          Operation mode, either "dry-run" or "scale-down"

          [default: dry-run]
          [possible values: scale-down, dry-run]

      --prometheus-url <PROMETHEUS_URL>
          Prometheus URL to query for GPU metrics eg. "http://prometheus-k8s.openshift-monitoring.svc:9090"

      --prometheus-token <PROMETHEUS_TOKEN>
          Prometheus token to use for authentication, if not provided, will try to authenticate using the service account token for the currently logged in OpenShift user

  -h, --help
          Print help (see a summary with '-h')
```


## TODOs

- emit a kubernetes event when a scaling action has occured