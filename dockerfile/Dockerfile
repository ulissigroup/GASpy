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
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y wget

# Install less
RUN apt-get install less

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
RUN echo -e "\nyes\n/miniconda3\nyes\n" | /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh
RUN rm Miniconda3-4.5.4-Linux-x86_64.sh $HOME/.bashrc-miniconda3.bak
ENV PATH /miniconda3/bin:$PATH

# Install conda packages
RUN conda config --prepend channels conda-forge
RUN conda config --append channels matsci
RUN conda install \
    numpy=1.15.1    scipy=1.1.0 \
    pytest=3.8.0 \
    mongodb=4.0.2   pymongo=3.7.1 \
    multiprocess=0.70.5 \
    ase=3.17.0 \
    pymatgen=2018.9.1 \
    fireworks=1.7.2 \
    luigi=2.7.8 \
    statsmodels=0.9.0 \
    ipykernel=4.9.0 \
    tqdm=4.24.0

# The $GASPY_HOME mount is so that you can use whatever version of GASpy.
# We do this near the end because we can't modify mounted folders after
# declaring them as volumes.
VOLUME $GASPY_HOME

# Make the default user `user` instead of `root`. Necessary when working with Shifter.
RUN chown -R user:group $HOME
USER user
