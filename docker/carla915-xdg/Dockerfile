FROM carlasim/carla:0.9.15

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      xdg-user-dirs \
    && rm -rf /var/lib/apt/lists/*

USER carla
