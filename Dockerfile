FROM continuumio/miniconda3

EXPOSE 8892/tcp

ENV CODE_DIR=/opt/code
ENV CONDA_ENV_NAME="phenoplier"
ENV N_JOBS=1
ENV ROOT_DIR=/opt/data
ENV USER_HOME=${ROOT_DIR}/user_home
ENV MANUSCRIPT_DIR=/opt/manuscript

VOLUME ${ROOT_DIR}
VOLUME ${MANUSCRIPT_DIR}

# install gnu parallel
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends parallel \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# setup phenoplier
COPY environment/environment.yml environment/scripts/install_other_packages.sh environment/scripts/install_r_packages.r /tmp/
RUN conda env create --name ${CONDA_ENV_NAME} --file /tmp/environment.yml \
  && conda run -n ${CONDA_ENV_NAME} --no-capture-output /bin/bash /tmp/install_other_packages.sh \
  && conda clean --all --yes

# activate the environment when starting bash
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=${CODE_DIR}/libs:${PYTHONPATH}

RUN echo "Make sure packages can be loaded"
RUN python -c "import papermill"

COPY . ${CODE_DIR}
WORKDIR ${CODE_DIR}

RUN echo "Make sure modules can be loaded"
RUN python -c "import conf; assert hasattr(conf, 'GENERAL')"

# setup user home directory
RUN mkdir ${USER_HOME} && chmod -R 0777 ${USER_HOME}
ENV HOME=${USER_HOME}

ENTRYPOINT ["/opt/code/entrypoint.sh"]
CMD ["scripts/run_nbs_server.sh", "--container-mode"]
