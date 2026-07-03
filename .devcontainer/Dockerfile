FROM python:3.12-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        antiword \
        build-essential \
        ca-certificates \
        curl \
        git \
        poppler-utils \
        ripgrep \
        sudo \
        tesseract-ocr \
        tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/* \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Codex CLI is installed at image build time so the container is ready to use.
# If the package name changes upstream, remove this line and install the current
# CLI package manually inside the container.
RUN npm install -g @openai/codex

USER ${USERNAME}
WORKDIR /workspaces/llm_law_rag
