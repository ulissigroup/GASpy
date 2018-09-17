# GASpy:  Generalized Adsorption Simulator for Python

We use Density Functional Theory (DFT) to calculate adsorption energies of
adsorbates onto slabs, but we try to do it in a general way such that we may
perform these calculations for an arbitrary number of configurations---e.g.,
different bulk materials, facets, adsorption sites, adsorbates, etc.

# Overview

GASpy is written in [Python 3](https://www.python.org/), and we use various tools
(below) that enable to perform DFT relaxations on sample pools of
near-arbitrary size. You can find a full list of our dependencies in our
[Dockerfile](https://github.com/ulissigroup/GASpy/blob/master/dockerfile/Dockerfile).

[ASE](https://wiki.fysik.dtu.dk/ase/about.html),
[VASP](https://www.vasp.at/index.php/about-vasp/59-about-vasp),
[PyMatGen](http://pymatgen.org/),
[FireWorks](https://pythonhosted.org/FireWorks/index.html), [Materials
Project](https://materialsproject.org/), [Docker](https://www.docker.com/),
[Luigi](https://github.com/spotify/luigi), [MongoDB](https://www.mongodb.com/)

We created various Python classes (referred to as
[tasks](https://github.com/ulissigroup/GASpy/tree/master/gaspy/tasks) by
[Luigi](https://github.com/spotify/luigi)) to automate adsorption energy
calculations. For example:  We have a task to fetch bulk structures from the
Materials Projec and then turn them into ASE atoms objects; we have a task that
uses PyMatGen to cut slabs out of these bulk structures; we have a task that
use PyMatGen to enumerate adsorption sites on these slabs; we have a task to
add adsorbates onto these sites; and we have a task to calculate adsorption
energies from slab, adsorbate, and slab+adsorbate relaxations.

We use Luigi to manage the dependencies between these tasks (e.g., our slab
cutting class requires that we have already fetched the bulk structure), and we
use FireWorks to manage/submit our DFT simulations across multiple clusters.
Thus, we can simply tell GASpy to "calculate the adsorption energy of CO on
Pt", and it automatically performs all of the necessary steps (e.g., fetch Pt
from Materials Project; cut slabs; relax the slabs; add CO onto the slab and
then relax; then calculate the adsorption energy).

To submit calculations, we create wrapper tasks that call on the appropriate
sub-tasks, and then use either Luigi our our Python wrapping functions to
execute the classes that you made. For
[example](https://github.com/ulissigroup/GASpy/blob/master/examples/calculate_all_adsorptions_on_surfaces.py):

    from gaspy.tasks import run_tasks
    from gaspy.tasks.submit_calculations.adsorption_calculations import AllSitesOnSurfaces
    
    
    rocket_builder = AllSitesOnSurfaces(adsorbates_list=[['CO'], ['H']],
                                        mpid_list=['mp-2', 'mp-30'],
                                        miller_list=[[1, 1, 1], [1, 0, 0]],
                                        max_rockets=20)
    tasks = [rocket_builder]
    
    run_tasks(tasks)

This snippet will calculate up to 20 CO and H adsorption energies of sites on
the (1, 1, 1) and (1, 0, 0) facets of
[Pt](https://materialsproject.org/materials/mp-2/) and
[Cu](https://materialsproject.org/materials/mp-30/).

# Installation

You will need five main things to run GASpy:

1. a locally cloned version of this repository,

2. [Docker](https://www.docker.com/),

3. a [MongoDB](https://www.mongodb.com/) server,

4. [FireWorks](https://pythonhosted.org/FireWorks/index.html) set up on your
   computing cluster(s), and

5. A properly configured
   [`.gaspyrc.json`](https://github.com/ulissigroup/GASpy/blob/master/.gaspyrc_template.json)
   file placed in your local GASpy folder.

## Docker

Our
image---[ulissigroup/gaspy](https://hub.docker.com/r/ulissigroup/gaspy/)---contains
the infrastructure that we use to run our code. Note that [this
image](https://github.com/ulissigroup/GASpy/blob/master/dockerfile/Dockerfile)
does not actually contain the GASpy source code. If it did, we would need to
constantly rebuild the image, because we are constantly changing and
redeveloping GASpy. We instead mount our local repository to the container that
we use to run our code: `docker run -v "/local/path/to/GASpy:/home/GASpy"
ulissigroup/gaspy:latest foo` You can also see how we open an interactive
Docker container
[here](https://github.com/ulissigroup/GASpy/blob/master/open_container_via_docker.sh).

## MongoDB

You will need to set up your own Mongo database and then put the appropriate
information into your
[`.gaspyrc.json`](https://github.com/ulissigroup/GASpy/blob/master/.gaspyrc_template.json)
file. You will need to make an `atoms` collection in your database, which will
contain one document for every DFT calculation you run. You will also need an
`adsorption` collection that will contain one document for every adsorption
energy you calculate, and a `relaxed_bulk_catalog` collection that will contain
one document for every adsorption site you enumerate using [this
script](https://github.com/ulissigroup/GASpy/blob/master/examples/enumerate_dft_catalog_manually.py).

We also have read-only mirrors to our catalog collections that allow for faster
reading. If you do not want to set this up, simply re-enter your catalog
collection's information into the readonly sections of the `.gaspyrc.json`
file.

The `surface_energy` collection is still under development; use at your
own risk.

## FireWorks

GASpy only submits jobs to
[FireWorks](https://materialsproject.github.io/fireworks/). You will need to
set up your own FireWorks database and rocket launchers on your computing
clusters. You will also need to enter the appropriate FireWorks data into the
`lpad*` sections of your
[`gaspyrc.json`](https://materialsproject.github.io/fireworks/) file.

## .gaspyrc.json

In addition to the aforementioned items you need to populate in your
`.gaspyrc.json` file, you will also need to set up a few other things:

- A dedicated folder to store the pickle files that Luigi will use to manage
  the task dependencies. This folder should be put it the `gasdb_path` field.
- A constantly running Luigi daemon. You can do this by simply running `nohup
  docker run -v "/local/path/to/GASpy:/home/GASpy" ulissigorup/gaspy:latest
  /miniconda3/bin/luigid &`. Then you enter the IP address of the machine that
  you ran that command on into the `luigi_host` field.
- You will need to get an API key from [The Materials
  Project](https://materialsproject.org/) and then enter it into the
  `matproj_api_key` field

You may notice the `gasdb_server` field. We use that to interface with a
web-based data viewing service that we still have under development. You will
not need to populate this field.

# Submodules

You may notice that we have two submodules:
[GASpy\_regressions](https://github.com/ulissigroup/GASpy_regressions) and
[GASpy\_feedback](https://github.com/ulissigroup/GASpy_feedback). We use our
regression submodule to analyze and perform regressions on our DFT data, and we
use our feedback submodule to choose which calculations to \[automatically\]
perform next.

# Reference

[Active learning across intermetallics to guide discovery of electrocatalysts
for CO2 reduction and H2
evolution](https://www.nature.com/articles/s41929-018-0142-1). Note that the
repository which we reference in this paper is version 0.1 of GASpy, which can
stil be found [here](https://github.com/ulissigroup/GASpy/tree/v0.1).

# Versions

Current GASpy version: 0.20

For an up-to-date list of our software dependencies, you can simply check out
how we build our docker image
[here](https://github.com/ulissigroup/GASpy/blob/master/dockerfile/Dockerfile).
