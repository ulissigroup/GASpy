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

# Add some things to the .bashrc file to make life easier.
COPY bashrc_additions.sh .
RUN cat bashrc_additions.sh >> /home/.bashrc
RUN rm bashrc_additions.sh

# Install GASpy. Note that we install it by assuming that the user will mount
# their working version of GASpy to the container.
ENV GASPY_HOME=$HOME/GASpy
RUN mkdir -p $GASPY_HOME
ENV PYTHONPATH $GASPY_HOME

# Install packages that we need to install other packages
RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get update && apt-get install -y less wget rsync git

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh --directory-prefix=$HOME
RUN /bin/bash $HOME/Miniconda3-4.7.12-Linux-x86_64.sh -bp /miniconda3
RUN rm $HOME/Miniconda3-4.7.12-Linux-x86_64.sh
ENV PATH /miniconda3/bin:$PATH

# Install conda packages
# note: default python version for conda 4.7.12 is 3.7
# which conflicts with many packages, used 3.6 instead
RUN conda config --prepend channels conda-forge
RUN conda config --append channels matsci
RUN conda install \
    python=3.6 \
    numpy=1.17.2    scipy>=1.1.0   pandas=0.25.1 \
    multiprocess>=0.70.5 \
    pytest=5.0.1 \
    mongodb>=4.0.2   pymongo=3.8.0 \
    ase=3.17.0 \
    pymatgen=2019.12.3  fireworks=1.7.2 \
    luigi>=2.8.9 \
    statsmodels>=0.9.0 \
    jupyter>=1.0.0  tqdm>=4.24.0
RUN conda clean -ity

# Only the development version of ipycache works right now
RUN pip install git+https://github.com/rossant/ipycache.git

# The $HOME/.ssh mount is so that you can mount your ~/.ssh into it, so that
# you can freely log into other things from inside the container.
# The $GASPY_HOME mount is so that you can use whatever version of GASpy.
# We do this near the end because we can't modify mounted folders after
# declaring them as volumes.
RUN mkdir -p $HOME/.ssh
VOLUME $HOME/.ssh $GASPY_HOME

# Make the default user `user` instead of `root`. Necessary when working with Shifter.
RUN chown -R user:group $HOME
USER user
