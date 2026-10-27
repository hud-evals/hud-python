FROM debian:bookworm-slim
WORKDIR /app
ENV IMAGE_ONLY=present
COPY entrypoint.sh /usr/local/bin/start-harbor-environment
RUN chmod +x /usr/local/bin/start-harbor-environment
ENTRYPOINT ["/usr/local/bin/start-harbor-environment"]
