GASpy installation is non-trivial.
It took GASpy developers ~1 week to do the last full installation from scratch, so new users may expect longer installation times.
Please contact us with any questions, comments, or issues.


# Overview

GASpy is generally run via an administrative [Docker](https://www.docker.com/) container.
This container will delegate the calculations out to your various computing clusters.
Here is what you need to set up for the administrative container:

1. A locally [cloned](https://help.github.com/en/articles/cloning-a-repository) version of this repository

2. Something to open [Docker](https://www.docker.com/) images such as:  Docker itself, [Shifter](https://github.com/NERSC/shifter), or [Singularity](https://singularity.lbl.gov/). If none of these are available to you, then you might be able to simply create your own [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment with the correct [dependencies](../dockerfile/Dockerfile)

3. A [MongoDB](https://www.mongodb.com/) server

4. A properly configured [`.gaspyrc.json`](../.gaspyrc_template.json) file placed in your local GASpy folder

5. A [Luigi](https://github.com/spotify/luigi) [daemon](https://luigi.readthedocs.io/en/stable/central_scheduler.html) running. This daemon does not have to be running inside a container. You can have it simply running on a barebones miniconda environment with only Luigi installed.

6. You may need [SSH tunnels](https://www.ssh.com/ssh/tunneling/) to allow your computing clusters to communicate with your FireWorks and/or MongoDB servers.

7. A series of [cron](https://en.wikipedia.org/wiki/Cron) jobs to automatically perform operations for you (details below).

Here is what you need to set up for the computing clusters you plan to use:

1. Working installations of the DFT calculator you want to use. GASpy is currently able to interface with [VASP](https://www.vasp.at/) and [Quantum Espresso](https://www.quantum-espresso.org/).

2. An environment with [FireWorks](https://pythonhosted.org/FireWorks/index.html) and [ASE](https://wiki.fysik.dtu.dk/ase/)

3. Some sort of wrapping tool that can accept run-parameters from GASpy and then run the DFT. See below for more details.

It can also use [RISM](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.115429), but this is still under development and getting it compiled is non-trivial.
If you want to use GASpy-RISM, please contact us.


# DFT wrapper

Our original VASP wrapper is inside [`gaspy.vasp_functions`](../gaspy/vasp_functions.py).
This is contains the scripts that we use to run VASP on our computing clusters.
You will find that they are hard-coded for our clusters.
There is no way to get around this.
You will need to make and substitute your own wrapper.
[Here](https://github.com/ulissigroup/espresso_tools) is an example of another wrapper we have for Quantum Espresso.
Again, this is hard-coded for our purposes.

Depending on how your wrapper is set up, you may need to fork GASpy and add/modify some scripts inside [`gaspy.fireworks_helper_scripts`](../gaspy/fireworks_helper_scripts.py).
Please contact us and/or send a PR if you need to do this.


# Docker

Our image---[ulissigroup/gaspy](https://hub.docker.com/r/ulissigroup/gaspy/)---contains the infrastructure that we use to run our code.
Note that this image does not actually contain the GASpy source code.
If it did, we would need to constantly rebuild the image, because we are constantly changing and redeveloping GASpy.
We instead mount our local repository to the container that we use to run our code:  `docker run -v "/local/path/to/GASpy:/home/GASpy" ulissigroup/gaspy:latest foo`.
You can also see how we open an interactive Docker container [here](../open_container_via_docker.sh).

Note that you only need to work within this container to submit and analyze calculations.
You do not need this container to run the calculations on each of your computing clusters.


# MongoDB

You will need to set up your own Mongo database and then put the appropriate information into your [`.gaspyrc.json`](../.gaspyrc_template.json) file.
You will need to make the following collections in your database:

- `atoms` will contain one document for every DFT calculation you run.
- `catalog` will contain one document for every adsorption site you [enumerate](../examples/populate_catalog.py).
- `adsorption` will contain one document for every adsorption energy you calculate.
- `surface_energy` will contain one document for every surface energy you calculate.

We also have read-only mirrors to our catalog collections that allow for faster reading.
If you do not want to set this up, simply re-enter your catalog collection's information into the readonly sections of the `.gaspyrc.json` file.


# FireWorks

Please refer to [Fireworks'](https://materialsproject.github.io/fireworks/) [installation instructions](https://materialsproject.github.io/fireworks/installation.html) for how to get your FireWorks database, launchpad, and qadapters set up on each of your computing clusters.


# .gaspyrc.json

In addition to the aforementioned items you need to populate in your `.gaspyrc.json` file, you will also need to set up a few other things:

- `temp_directory`:  Some sort of folder that will be used to write files temporarily. This will likely be `/tmp/`. GASpy uses this directory to perform [atomic writing operations](https://en.wikipedia.org/wiki/Atomicity_(database_systems)).
- `luigi_host`:  You need to set up a constantly running Luigi daemon. You can do this by running `nohup docker run -v "/local/path/to/GASpy:/home/GASpy" ulissigorup/gaspy:latest /miniconda3/bin/luigid &`. Then you enter the IP address of the machine that you ran that command on into the `luigi_host` field.
- `gasdb_path`:  A dedicated folder to store various cache files, including the [pickles](https://docs.python.org/3/library/pickle.html) that Luigi will use to manage the task dependencies. If you have a [scratch directory](https://en.wikipedia.org/wiki/Scratch_space) on your computing cluster, then it would be safe to put this pickle file in there.
- `matproj_api_key`:  You will need to get an API key from [The Materials Project](https://materialsproject.org/) and then enter it into this field. This will let you fetch bulk materials from their database, which will be the seed for all of GASpy's calculations.
- `dft_calculator`:  The default DFT calculator you want to use. Currently supports values of `vasp`, `qe`, and `rism`. Note that this only supplies the default. You will still be able to run VASP and Quantum Espresso concurrently if you want to.
- `plotly_login_info`:  Our `../GASpy_regressions` submodule uses this field to push our analyses into public websites. If you only plan to use GASpy to run calculations, then you will not need to populate this field.
- `gasdb_server`:  This field is something we use to interface with a web-based data viewing service that we still have under development. It is still experimental, and you will not need to populate it.
- `fireworks_info`:  You will need to point to your FireWorks `launchpad.yaml` file, as well as enter the login information for your FireWorks Mongo server.
- `mongo_info`:  You will need to add the login information for your Mongo collections here. Note that they are separated so that you can technically have different Mongo servers hosting the different collections.


# cron

## Luigi daemon
As mentioned previously, you will need to have a Luigi daemon constantly running.
In our experience, it is common for these daemons to shut down for various reasons (server creashes, etc.)
To address this issue, we have cron jobs that periodically check the status of the daemon and then restart the daemon if needed.
For example:  `0 0 * * * bash -c "if [ ! $(ps xw | grep luigid | grep -vq grep) ]; then luigid; fi"`.
Note that this is only a template for an idea.
You can still handle this however you would like, and details (e.g., getting the right environment to start Luigi) are left up to you.

## GASdb
Once you submit a job to GASpy, it will be queued to run in FireWorks.
But it will not be automatically added to your database.
For this to happen automatically, you will need to execute the `../examples/update_collections.py` script periodically.
This script will need to be executed inside a GASpy container though.
This means that you will need to figure out how to run that script through whichever Docker image solution you choose, and then you will need to put it into a cron job.

## FireWorks rocket launchers
On each of your computing clusters, you should be able to use FireWorks to submit jobs to the queue [continuously](https://materialsproject.github.io/fireworks/queue_tutorial.html#continually-submit-jobs-to-the-queue).
But like the Luigi daemon, our experience has showed us that these launches sometimes crash.
We have cron jobs to restart them if needed, e.g., `0 0 * * * bash -c "if [ ! $(ps xw | grep fireworks | grep -vq grep) ]; then qlaunch --launchpad_file=foo --fworker_file=bar --queueadapter_file=foobar rapidfire -m 20 --nlaunches infinite >> logfile 2>&1; fi"`

## Tunnels
As we stated before, each of your computing clusters may need to have SSH tunnels to connect tothe FireWorks database.
Just like the Luigi daemon and the FireWorks launches, we have cron jobs to check that the tunnels are still up, e.g., `0 0 * * * bash -c "if [ ! $(ps xw | grep cluster_name | grep -vq grep) ]; then ssh -gnNT -L 0.0.0.0:30000:mongo.server:27017 cluster.name; fi"`


# Unit testing

There is a lot of configuration and overhead set up and it is very easy to miss something.
The best way to make sure that you've configured everything correctly is to perform GASpy's unit tests.
You can do this by opening a container [via Docker](../open_container_via_docker.sh), [via Shifter](../open_container_via_shifter.sh), or via Singularity (contact us for information regarding Singularity).
Then you need to `cd /home/GASpy/gaspy/tests` and execute `pytest`.
This will call [pytest](https://docs.pytest.org/en/latest/) to run all of the unit tests.
From there you will need to figure out what is failing and why.
If you need assistance, please do not hesitate to contact us.
