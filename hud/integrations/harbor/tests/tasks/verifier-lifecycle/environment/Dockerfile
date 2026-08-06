FROM debian:bookworm-slim
WORKDIR /app
COPY entrypoint.sh /usr/local/bin/start-verifier-fixture
RUN chmod +x /usr/local/bin/start-verifier-fixture
ENTRYPOINT ["/usr/local/bin/start-verifier-fixture"]
