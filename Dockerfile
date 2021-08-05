FROM continuumio/miniconda3

EXPOSE 8892/tcp

ENV PHENOPLIER_CODE_DIR=/opt/phenoplier_code
ENV PHENOPLIER_N_JOBS=1
ENV PHENOPLIER_ROOT_DIR=/opt/phenoplier_data
ENV PHENOPLIER_MANUSCRIPT_DIR=/opt/phenoplier_manuscript

VOLUME ${PHENOPLIER_ROOT_DIR}
VOLUME ${PHENOPLIER_MANUSCRIPT_DIR}

# setup phenoplier
COPY environment/environment.yml /tmp/
RUN conda env create --name phenoplier --file /tmp/environment.yml
COPY environment/scripts/install_other_packages.sh environment/scripts/install_r_packages.r /tmp/
RUN ["conda", "run", "-n", "phenoplier", "--no-capture-output", "/bin/bash", "/tmp/install_other_packages.sh"]

RUN echo "conda activate phenoplier" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=${PHENOPLIER_CODE_DIR}/libs:${PYTHONPATH}

RUN echo "Make sure packages can be loaded"
RUN python -c "import rpy2.robjects as ro"

COPY . ${PHENOPLIER_CODE_DIR}
WORKDIR ${PHENOPLIER_CODE_DIR}

RUN echo "Make sure modules can be loaded"
RUN python -c "import conf"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["scripts/run_nbs_server.sh"]

