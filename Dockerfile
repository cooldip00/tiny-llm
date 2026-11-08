# The tinyllm server on CPU (6.5b, D-033): the image behind the Hugging Face Space, and a local `docker run`.
#
#   docker build -t tinyllm --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
#   docker run --rm -p 7860:7860 tinyllm
#   curl localhost:7860/v1/completions -H 'content-type: application/json' \
#        -d '{"prompt": "Once upon a time", "max_tokens": 60}'
#
# CPU only: no CUDA, Triton or CUDA graphs here, so the model runs the PyTorch code paths
# (/stats reports norm_backend "torch"). The speed numbers in the README come from the laptop GPU.
# No weights in the image: the model is downloaded from the Hub at start into the HF cache
# (MODEL / REVISION below), or mounted as a folder: -v /path/to/export:/model -e MODEL=/model.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU build of the torch version in pyproject.toml (as in CI), then only what the server imports.
# Versions match uv.lock; training-only packages (datasets, wandb) stay out of the image.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy==2.5.4 tokenizers==0.23.3 huggingface-hub==2.4.0 \
                fastapi==0.162.0 starlette==1.10.0 pydantic==2.14.2 uvicorn==0.54.0 \
 && python -c "import torch; assert not torch.cuda.is_available() and torch.version.cuda is None"

# Spaces run the container as uid 1000; the HF cache must be writable by it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONPATH=/app/src
WORKDIR /app

COPY --chown=user src/tinyllm ./src/tinyllm
COPY --chown=user tokenizer ./tokenizer

ARG GIT_COMMIT=unknown
# REVISION pins the Hub commit of the model upload (scripts/upload_model.py), so a rebuild serves the same weights.
ENV TINYLLM_GIT_COMMIT=$GIT_COMMIT \
    MODEL=CoolDip00/tiny-llm-tinystories-30m REVISION=b11100b189f9d5085ce8c0fd775ca71a6eaf287c \
    SLOTS=8 PORT=7860 SLO_TTFT_S=2 LOG_DATASET="" \
    MAX_QUEUE=4 RATE_LIMIT=20 RATE_BURST=5 PROXY_HOPS=1 TRUST_ROUTER=""

EXPOSE 7860
# Shell form so every ENV above can be overridden with -e (or a Space variable); exec keeps uvicorn as
# PID 1 (SIGTERM → graceful drain, then the last log upload). --revision only for a repo id: a mounted
# folder has none. SLO_TTFT_S: the status page's TTFT SLO threshold, 2 s on the shared CPU.
# MAX_QUEUE / RATE_LIMIT (requests/min per visitor, 0 = off) / RATE_BURST: load shedding (docs/SERVING.md).
# SLOTS / MAX_QUEUE / RATE_*: from the 2-CPU load test (docs/SERVING.md "Space defaults"): 8 slots,
# queue ≈ 2 s TTFT target × throughput, 20 requests/min per visitor with a burst of 5.
# --trust-proxy only on a Space (HF sets SPACE_ID): there HF's proxy adds the visitor's address to
# X-Forwarded-For; a plain `docker run` has no proxy, so the header is the client's own and is ignored.
# PROXY_HOPS: proxies in front that append to that header (the client is that many entries from the right).
# TRUST_ROUTER (off when empty): only on a private replica behind the router (D-031). Never on a public
# Space: a visitor could send the header themselves and get a new identity for each request.
# LOG_DATASET (off when empty): a private HF dataset for the request logs (docs/SERVING.md "Privacy");
# needs HF_TOKEN with write access to it, and TINYLLM_LOG_SALT to count visitors across restarts.
CMD exec python -m tinyllm.server --hf-model "$MODEL" \
    $( [ -d "$MODEL" ] || echo --revision "$REVISION" ) \
    $( [ -z "$LOG_DATASET" ] || echo --log-dir /tmp/tinyllm-logs --log-dataset "$LOG_DATASET" ) \
    $( [ -z "$SPACE_ID" ] || echo --trust-proxy --proxy-hops "$PROXY_HOPS" ) \
    $( [ -z "$TRUST_ROUTER" ] || echo --trust-router ) \
    --device cpu --host 0.0.0.0 --port "$PORT" --slots "$SLOTS" --slo-ttft-s "$SLO_TTFT_S" \
    --max-queue "$MAX_QUEUE" --rate-limit "$RATE_LIMIT" --rate-burst "$RATE_BURST"
