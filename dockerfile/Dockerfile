FROM ubuntu:18.04
LABEL maintainers.owner="Zack Ulissi <zulissi@andrew.cmu.edu>"
LABEL maintainers.maintainer0="Kevin Tran <ktran@andrew.cmu.edu>"
SHELL ["/bin/bash", "-c"]

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
VOLUME /root/GASpy
ENV PYTHONPATH "/root/GASpy/GASpy_feedback:$PYTHONPATH"
ENV PYTHONPATH "/root/GASpy/GASpy_regressions:$PYTHONPATH"
ENV PYTHONPATH "/root/GASpy:$PYTHONPATH"

# Install packages that we need to install other packages
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y \
    wget \
    git

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
RUN echo -e "yes\n\yes\nyes\n" | /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh
RUN rm Miniconda3-4.5.4-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin/:$PATH
# Install Python
RUN conda create -n GASpy_conda python
RUN echo "source activate GASpy_conda" >> /root/.bashrc
RUN source activate GASpy_conda && pip install pytest

# Install some dependencies for matplotlib that need to be installed via apt-get
RUN apt-get install -y \
    pkg-config \
    libfreetype6-dev \
    libpng-dev

# Install conda packages. First we set the default channel to conda-forge
RUN source activate GASpy_conda && conda config --prepend channels conda-forge
# Foundational packages
RUN source activate GASpy_conda && conda install \
    numpy \
    scipy
# Data management packages
RUN source activate GASpy_conda && conda install \
    pandas \
    dill \
    gspread \
    pymongo
# Visualization packages
RUN source activate GASpy_conda && conda install \
    matplotlib \
    seaborn \
    plotly
# CS packages
RUN source activate GASpy_conda && conda install \
    multiprocess \
    jupyter \
    luigi
# Scientific packages
RUN source activate GASpy_conda && conda install \
    scikit-learn \
    ase
# LBL packages
RUN source activate GASpy_conda && conda install -c matsci \
    fireworks \
    pymatgen

# Pip packages
RUN source activate GASpy_conda && pip install tpot

# Make sure numpy is up-to-date. This is a hack to fix some building errors.
RUN source activate GASpy_conda && conda update numpy

# Open up Jupyter during startup
# This is optional, and the image should be built as `gaspy_jupyter` instead of `gaspy`
#COPY open_jupyter.sh /root/open_jupyter.sh
#CMD ["/root/open_jupyter.sh"]
