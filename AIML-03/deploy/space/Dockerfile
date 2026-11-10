# Hugging Face Space (Docker SDK) for the demo. Builds from the public GitHub repo and the trimmed model bundle branch.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2 \
    BUNDLE=/home/user/app/bundle PORT=7860
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/ram15x/personalized-recommendation-engine.git /tmp/repo \
 && cp -r /tmp/repo/src /tmp/repo/demo /tmp/repo/requirements-space.txt . && rm -rf /tmp/repo
RUN pip install --no-cache-dir -r requirements-space.txt
RUN git clone --depth 1 --branch demo-bundle https://github.com/ram15x/personalized-recommendation-engine.git bundle \
 && rm -rf bundle/.git bundle/README.md

EXPOSE 7860
CMD ["python", "demo/app.py"]
