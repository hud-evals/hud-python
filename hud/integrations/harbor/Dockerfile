ARG BASE_IMAGE
ARG VERIFIER_IMAGE

FROM ${BASE_IMAGE} AS authored-root
USER root
COPY install.sh /tmp/hud-install.sh
RUN sh /tmp/hud-install.sh --system-only && rm /tmp/hud-install.sh

FROM ${VERIFIER_IMAGE} AS verifier-root
FROM docker:28.3.3-cli AS docker-cli
FROM python:3.12-slim AS runtime

USER root
ARG HUD_REQUIREMENT=hud
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /controller/bin/uv
COPY install.sh /controller/
COPY packages /controller/packages
RUN sh /controller/install.sh "${HUD_REQUIREMENT}" && mkdir -p /runtime
# Task-specific inputs follow the install, so every task reuses its layer.
COPY config.json /controller/
COPY --from=authored-root / /rootfs

ENV HUD_SKIP_VERSION_CHECK=1
EXPOSE 8765
ENTRYPOINT []
CMD ["/controller/venv/bin/hud", "serve", "hud.integrations.harbor.env:env", "--host", "0.0.0.0", "--port", "8765"]

FROM runtime AS plain

FROM runtime AS service-access
COPY --from=docker-cli /usr/local/bin/docker /controller/bin/docker

FROM service-access AS verifier
COPY --from=verifier-root / /verifier
