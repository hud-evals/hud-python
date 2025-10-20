FROM quay.io/jupyter/minimal-notebook:python-3.11

USER root

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MCP_TRANSPORT="stdio" \
    HUD_LOG_STREAM="stderr"

WORKDIR /app

# Step 1: Copy dependency files for caching
COPY server/pyproject.toml /app/server/
COPY environment/pyproject.toml /app/environment/

# Step 2: Install all dependencies
RUN pip install --no-cache-dir -e /app/server/ -e /app/environment/

# Step 3: Copy code
COPY server/ /app/server/
COPY environment/ /app/environment/
COPY notebooks/ /app/notebooks/

# Create directories with proper ownership
RUN mkdir -p /app/data /app/logs /app/shared_data /app/workspace && \
    chown -R $NB_UID:$NB_GID /app

USER $NB_UID

# Step 4: Download and extract data from HuggingFace (as non-root user)
RUN wget -q https://huggingface.co/datasets/KAKA22/SpreadsheetBench/resolve/main/all_data_912.tar.gz -O /tmp/data.tar.gz && \
    tar --no-same-permissions -xzf /tmp/data.tar.gz -C /app/data && \
    rm /tmp/data.tar.gz

EXPOSE 8000 8888

CMD ["sh", "-c", "\
    jupyter kernelgateway --KernelGatewayApp.ip=0.0.0.0 --KernelGatewayApp.port=8888 >&2 & \
    uvicorn environment.server:app --host 0.0.0.0 --port 8000 --log-level warning >&2 & \
    sleep 5 && cd /app/server && exec python3 -m server.main\
    "]
