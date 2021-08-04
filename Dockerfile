FROM ubuntu:20.04

ENV PHENOPLIER_CODE_DIR=/opt/phenoplier_code
ENV PHENOPLIER_N_JOBS=1
ENV PHENOPLIER_ROOT_DIR=/opt/phenoplier_data
ENV PHENOPLIER_MANUSCRIPT_DIR=/opt/phenoplier_manuscript

VOLUME ${PHENOPLIER_ROOT_DIR}
VOLUME ${PHENOPLIER_MANUSCRIPT_DIR}

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O miniconda3.sh \
    && mkdir /root/.conda \
    && bash miniconda3.sh -b \
    && rm -f miniconda3.sh

# setup phenoplier
COPY . ${PHENOPLIER_CODE_DIR}
RUN cd ${PHENOPLIER_CODE_DIR}/environment \
    && conda env create --name phenoplier --file environment.yml

# Make RUN commands use the new environment
RUN echo "conda activate phenoplier" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

WORKDIR ${PHENOPLIER_CODE_DIR}

RUN ["conda", "run", "-n", "phenoplier", "--no-capture-output", "/bin/bash", "environment/scripts/install_other_packages.sh"]

#RUN cd ${PHENOPLIER_CODE_DIR}/environment \
#    && bash scripts/install_other_packages.sh

ENV PYTHONPATH=${PHENOPLIER_CODE_DIR}/libs

