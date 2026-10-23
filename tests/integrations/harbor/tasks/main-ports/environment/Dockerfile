FROM debian:bookworm-slim
WORKDIR /app
COPY entrypoint.sh /usr/local/bin/start-environment
RUN chmod +x /usr/local/bin/start-environment && printf nested > /app/marker
EXPOSE 3128 3129 8765 8080
ENTRYPOINT ["/usr/local/bin/start-environment"]
