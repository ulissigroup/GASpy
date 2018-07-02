FROM ubuntu:18.04
LABEL maintainers.owner="Zack Ulissi <zulissi@andrew.cmu.edu>"
LABEL maintainers.maintainer0="Kevin Tran <ktran@andrew.cmu.edu>"
SHELL ["/bin/bash", "-c"]

# Install packages that we need to install other packages
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y \
    wget \
    git

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
RUN echo -e "yes\nyes" | /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin/:$PATH
# Install Python
RUN conda create -n GASpy_conda python
RUN echo "source activate GASpy_conda" >> /root/.bashrc

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
VOLUME /home/GASpy
ENV PYTHONPATH "/home/GASpy/GASpy_feedback:$PYTHONPATH"
ENV PYTHONPATH "/home/GASpy/GASpy_regressions:$PYTHONPATH"
ENV PYTHONPATH "/home/GASpy:$PYTHONPATH"

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
# Scientific packages
RUN source activate GASpy_conda && conda install \
    scikit-learn \
    ase
# Install LBL packages
RUN source activate GASpy_conda && conda install -c matsci \
    fireworks \
    pymatgen

# Install pip packages
RUN source activate GASpy_conda && pip install git+git://github.com/jkitchin/vasp.git
RUN source activate GASpy_conda && pip install tpot

# Make sure numpy is up-to-date. This is a hack to fix some building errors.
RUN source activate GASpy_conda && conda update numpy
