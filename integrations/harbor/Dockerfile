ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
ARG HUD_REQUIREMENT=hud
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /media/hud/bin/uv
COPY env.py install.sh tasks.json /media/hud/
COPY tasks /media/hud/tasks
COPY packages /media/hud/packages
RUN sh /media/hud/install.sh "${HUD_REQUIREMENT}"

ENV HUD_SKIP_VERSION_CHECK=1
EXPOSE 8765
ENTRYPOINT []
CMD ["/media/hud/venv/bin/hud", "serve", "/media/hud/env.py", "--host", "0.0.0.0", "--port", "8765"]
