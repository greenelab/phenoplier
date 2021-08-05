FROM continuumio/miniconda3

EXPOSE 8892/tcp

ENV PHENOPLIER_CODE_DIR=/opt/phenoplier_code
ENV PHENOPLIER_N_JOBS=1
ENV PHENOPLIER_ROOT_DIR=/opt/phenoplier_data
ENV PHENOPLIER_MANUSCRIPT_DIR=/opt/phenoplier_manuscript

VOLUME ${PHENOPLIER_ROOT_DIR}
VOLUME ${PHENOPLIER_MANUSCRIPT_DIR}

# install gnu parallel
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends parallel \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# setup phenoplier
COPY environment/environment.yml environment/scripts/install_other_packages.sh environment/scripts/install_r_packages.r /tmp/
RUN conda env create --name phenoplier --file /tmp/environment.yml \
  && conda run -n phenoplier --no-capture-output /bin/bash /tmp/install_other_packages.sh \
  && conda clean --all --yes

# activate the environment when starting bash
RUN echo "conda activate phenoplier" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=${PHENOPLIER_CODE_DIR}/libs:${PYTHONPATH}

RUN echo "Make sure packages can be loaded"
RUN python -c "import rpy2.robjects as ro"

COPY . ${PHENOPLIER_CODE_DIR}
WORKDIR ${PHENOPLIER_CODE_DIR}

RUN echo "Make sure modules can be loaded"
RUN python -c "import conf"

ENTRYPOINT ["/opt/phenoplier_code/entrypoint.sh"]
CMD ["scripts/run_nbs_server.sh", "--container-mode"]

