FROM ubuntu:18.04
LABEL maintainers.owner="Zack Ulissi <zulissi@andrew.cmu.edu>"
LABEL maintainers.maintainer0="Kevin Tran <ktran@andrew.cmu.edu>"
SHELL ["/bin/bash", "-c"]

# Set up a non-root user, `user`, with a group, `group`
ENV HOME=/home
RUN mkdir -p $HOME
RUN groupadd -r group && \
    useradd -r -g group -d $HOME -s /sbin/nologin -c "Default user" user
RUN cp /root/.bashrc $HOME
RUN chown -R user:group $HOME

# Add some things to the .bashrc file to make life easier.
COPY bashrc_additions.sh .
RUN cat bashrc_additions.sh >> /home/.bashrc
RUN rm bashrc_additions.sh

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
ENV GASPY_HOME=$HOME/GASpy
RUN mkdir -p $GASPY_HOME
ENV PYTHONPATH $GASPY_HOME/GASpy_feedback:$PYTHONPATH
ENV PYTHONPATH $GASPY_HOME/GASpy_regressions:$PYTHONPATH
ENV PYTHONPATH $GASPY_HOME:$PYTHONPATH

# Install packages that we need to install other packages
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y \
    wget \
    git

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
RUN echo -e "\nyes\n/miniconda3\nyes\n" | /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh
RUN rm Miniconda3-4.5.4-Linux-x86_64.sh $HOME/.bashrc-miniconda3.bak
ENV PATH /miniconda3/bin:$PATH
# Install Python
RUN conda create -n GASpy_conda python
RUN echo "source activate GASpy_conda" >> $HOME/.bashrc

# Install conda packages. First we add the matsci and conda-forge channels
# to the conda, where conda-forge is the default
RUN source activate GASpy_conda && conda config --prepend channels matsci
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
RUN source activate GASpy_conda && pip install pytest
RUN source activate GASpy_conda && pip install tpot
RUN source activate GASpy_conda && pip install git+https://github.com/rossant/ipycache.git

# Make sure numpy is up-to-date. This is a hack to fix some building errors.
RUN source activate GASpy_conda && conda update numpy

# The $HOME mount is so that you can pass your `~/.ssh` folder
# into the container, which will allow you to ssh wherever.
# The $GASPY_HOME mount is so that you can use whatever version of GASpy.
# We do this near the end because we can't modify mounted folders after
# declaring them as volumes.
VOLUME $HOME $GASPY_HOME

# Open the tunnel to the Mongo server through a startup script.
# If the user passes the argument "jupyter" to the script, then Jupyter
# will be started. The user should connect to the 0.0.0.0:8888 port
# of the container to access the Jupyter interace.
COPY --chown=user:group startup.sh /home/startup.sh
ENTRYPOINT ["/home/startup.sh"]
CMD ["0"]

# Make the default user `user` instead of `root`
USER user
