set dotenv-load := true 

repository := "quay.io/wseaton"
project := "gpu-pruner"
version := `yq -oy ".workspace.package.version" Cargo.toml`

run:
  cargo run --release --bin {{ project }} -- -t 2880 --enabled-resources="n" --run-mode=dry-run --prometheus-url=$PROMETHEUS_URL

querytest:
  cargo run --release --bin querytest 'DCGM_FI_PROF_GR_ENGINE_ACTIVE{exported_namespace =~ "(.*)-nb"}[30d]' $PROMETHEUS_URL 


build-docker:
  podman build -t localhost/{{ project }}:{{ version }}-otel -f Dockerfile.rhel
  podman push localhost/{{ project }}:{{ version }}-otel {{ repository }}/{{ project }}:{{ version }}-otel 

build-docker-non-otel:
  podman build -t localhost/{{ project }}:{{ version }} -f Dockerfile.rhel --build-arg FEATURES="default"
  podman push localhost/{{ project }}:{{ version }} {{ repository }}/{{ project }}:{{ version }}


build-all : build-docker build-docker-non-otel