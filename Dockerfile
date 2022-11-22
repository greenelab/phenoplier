# syntax=docker/dockerfile:1

#
# base image: it contains the conda environment.
#
#  This image should be labeled as miltondp/phenoplier:base-latest when building,
#  so the "final image" below can use it.
#
FROM continuumio/miniconda3 AS base

EXPOSE 8892/tcp

ENV CODE_DIR=/opt/code
ENV CONDA_ENV_NAME="phenoplier"

ENV PHENOPLIER_N_JOBS=1
ENV PHENOPLIER_ROOT_DIR=/opt/data
ENV PHENOPLIER_USER_HOME=${PHENOPLIER_ROOT_DIR}/user_home
ENV PHENOPLIER_MANUSCRIPT_DIR=/opt/manuscript

VOLUME ${PHENOPLIER_ROOT_DIR}
VOLUME ${PHENOPLIER_MANUSCRIPT_DIR}

# install gnu parallel
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends parallel build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# setup phenoplier
COPY environment/environment.yml environment/scripts/install_other_packages.sh environment/scripts/install_r_packages.r /tmp/
RUN conda config --add channels conda-forge \
  && conda config --set channel_priority strict \
  && conda env create --name ${CONDA_ENV_NAME} --file /tmp/environment.yml \
  && conda run -n ${CONDA_ENV_NAME} --no-capture-output /bin/bash /tmp/install_other_packages.sh \
  && conda clean --all --yes \
  && chmod -R 0777 /opt/conda/envs/${CONDA_ENV_NAME} \
  && echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# activate the environment
SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=${CODE_DIR}/libs:${PYTHONPATH}

# setup user home directory
RUN mkdir -p ${PHENOPLIER_USER_HOME} && chmod -R 0777 ${PHENOPLIER_USER_HOME}
ENV HOME=${PHENOPLIER_USER_HOME}


#
# final image: it only copies the source code of the project into the image.
#
#  The idea is to avoid rebuilding the conda environment each time the source
#  code nees to be copied/updated.
#
FROM miltondp/phenoplier:base-latest AS final

COPY --chmod=777 . ${CODE_DIR}
RUN chmod 777 ${CODE_DIR}
WORKDIR ${CODE_DIR}

ENTRYPOINT ["/opt/code/entrypoint.sh"]
CMD ["scripts/run_nbs_server.sh", "--container-mode"]

