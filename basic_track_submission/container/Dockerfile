FROM ubuntu:22.04

LABEL org.opencontainers.image.title="LMM-in-AutoDrive Basic Track"
LABEL org.opencontainers.image.description="Bench2Drive-compatible unified three-scene sensor-policy VLA"
LABEL org.opencontainers.image.source="https://github.com/ZhuShanzhe/LMM-in-AutoDrive"
LABEL org.opencontainers.image.revision="4bec242"
LABEL org.opencontainers.image.bench2drive.revision="2645714eb1f3a100217928dd113093cae0779f36"
LABEL org.opencontainers.image.carla.version="0.9.16"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libfontconfig1 \
    libfreetype6 \
    libgl1 \
    libglib2.0-0 \
    libice6 \
    libjpeg8 \
    libpng16-16 \
    libsdl2-2.0-0 \
    libsm6 \
    libtiff5 \
    libvulkan1 \
    libx11-6 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    procps \
    unzip \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARLA_ROOT=/workspace/CARLA_0.9.16 \
    BENCH2DRIVE_ROOT=/workspace/Bench2Drive \
    REPOSITORY_ROOT=/workspace/LMM-in-AutoDrive \
    SCENARIO_RUNNER_ROOT=/workspace/Bench2Drive/scenario_runner \
    LEADERBOARD_ROOT=/workspace/Bench2Drive/leaderboard \
    TEAM_AGENT=/workspace/Bench2Drive/leaderboard/team_code/universal_vla_agent.py \
    TEAM_CONFIG=/workspace/submission/bench2drive/agent_config.json \
    CHALLENGE_TRACK_CODENAME=SENSORS \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    PATH=/root/autodl-tmp/conda_envs/command_parser/bin:${PATH} \
    PYTHONPATH=/workspace/CARLA_0.9.16/PythonAPI:/workspace/CARLA_0.9.16/PythonAPI/carla:/workspace/Bench2Drive/scenario_runner:/workspace/Bench2Drive/leaderboard:/workspace/Bench2Drive/leaderboard/team_code:/workspace/LMM-in-AutoDrive:/workspace/LMM-in-AutoDrive/experiment/CARLA

WORKDIR /workspace
COPY CARLA_0.9.16 /workspace/CARLA_0.9.16
COPY conda_env /root/autodl-tmp/conda_envs/command_parser
COPY LMM-in-AutoDrive /workspace/LMM-in-AutoDrive
COPY Bench2Drive /workspace/Bench2Drive
COPY models /workspace/models
COPY submission /workspace/submission

RUN chmod +x \
      /workspace/CARLA_0.9.16/CarlaUE4.sh \
      /workspace/submission/container/entrypoint.sh \
      /workspace/submission/container/run_bench2drive.sh \
    && mkdir -p /workspace/outputs /workspace/Bench2Drive/leaderboard/team_code \
    && cp /workspace/submission/bench2drive/team_code/universal_vla_agent.py \
      /workspace/Bench2Drive/leaderboard/team_code/universal_vla_agent.py \
    && ln -sf /root/autodl-tmp/conda_envs/command_parser/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /root/autodl-tmp/conda_envs/command_parser/bin/python3.12 /usr/local/bin/python \
    && python3.12 -m py_compile \
      /workspace/Bench2Drive/leaderboard/team_code/universal_vla_agent.py \
      /workspace/LMM-in-AutoDrive/experiment/CARLA/universal_vla_controller.py \
    && python3.12 -c "import torch; assert torch.__version__; print(torch.__version__)"

WORKDIR /workspace/LMM-in-AutoDrive
ENTRYPOINT ["/workspace/submission/container/entrypoint.sh"]
CMD ["bash"]
