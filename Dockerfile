FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
RUN uv pip install --system --no-cache --no-deps accelerate==1.15.0 runai-model-streamer==0.17.0 humanize==4.13.0
