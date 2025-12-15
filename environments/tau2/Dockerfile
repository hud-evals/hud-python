FROM python:3.12-slim

WORKDIR /app

# Download dataset and install tau2-bench from github
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/sierra-research/tau2-bench.git
RUN pip install -e tau2-bench

# Install other dependencies
COPY pyproject.toml ./pyproject.toml
RUN pip install -e .

# Copy source file
COPY server/ ./server/
COPY task/ ./task/

# Environment variable
ENV TAU2_DATA_DIR=/app/tau2-bench/data

CMD ["python", "-m", "server.main"]
