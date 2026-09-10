FROM carla:0.9.15-xdg

USER root

COPY AdditionalMaps_0.9.15.tar.gz /home/carla/Import/AdditionalMaps_0.9.15.tar.gz

RUN chown carla:carla /home/carla/Import/AdditionalMaps_0.9.15.tar.gz

USER carla
WORKDIR /home/carla

RUN ./ImportAssets.sh
