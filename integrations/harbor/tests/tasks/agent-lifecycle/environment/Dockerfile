FROM debian:bookworm-slim
RUN groupadd --gid 2000 worker && useradd --create-home --uid 1000 --gid 2000 agent
WORKDIR /app
RUN chown agent:worker /app
