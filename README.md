# GASpy:  Generalized Adsorption Simulator for Python

# Purpose
To use Density Functional Theory (DFT) to calculate adsorption energies of adsorbates onto slabs,
but to do it in a general way such that we may perform these calculations for an arbitrary
number of configurations (e.g., slab materials, adsorbate types, adsorption sites, percent
coverages, etc.).

# Overview
GASpy is written in Python, and we use various tools that enable us to begin DFT relaxations on
"arbitrarily large" sample pools.

## Tools
[ASE](https://wiki.fysik.dtu.dk/ase/about.html)

[PyMatGen](http://pymatgen.org/)

[Luigi](https://github.com/spotify/luigi)

[FireWorks](https://pythonhosted.org/FireWorks/index.html)

[VASP](https://www.vasp.at/index.php/about-vasp/59-about-vasp)

[VASP.py](https://github.com/jkitchin/vasp)

[Materials Project](https://materialsproject.org/)

[MongoDB](https://www.mongodb.com/)

## Infrastructure
We created various Python classes (AKA "tasks") to automate adsorption energy calculations;
these tasks live inside `gaspy/tasks.py`. For example:  We have a task to fetch bulk structures
from the Materials Projec and then turn them into ASE atoms objects; we have a task that uses
PyMatGen to cut slabs out of these bulk structures; we have a task that use PyMatGen to enumerate
adsorption sites on these slabs; we have a task to add adsorbates onto these sites; and we have
a task to calculate adsorption energies from slab, adsorbate, and slab+adsorbate relaxations.

We use Luigi to manage the dependencies between these tasks (e.g., our slab cutting class requires
that we have already fetched the bulk structure), and we use FireWorks to manage/submit our DFT
simulations across multiple clusters. Thus, we can simply tell GASpy to "calculate the adsorption
energy of CO on Pt", and it automatically performs all of the necessary steps (e.g., fetch Pt from
Materials Project; cut slabs; relax the slabs; add CO onto the slab and then relax; then calculate
the adsorption energy).

Note also that there is a `gaspy` package with various helper functions that may (and should) be
used in tasks, where appropriate. For example:  `gaspy.gasdb.get_docs` is useful for
pulling mongo documents from our database.

# Use
We recommend using [submodules](https://git-scm.com/docs/git-submodule) to create/execute your own custom
tasks or to analyze data that is created by GASpy. If you are interested in seeing a submodule but
do not have access, simply request access from a GASpy owner.

To submit calculations, create Luigi tasks that call on the appropriate tasks in `tasks.py`,
and then use Luigi to execute the classes that you made. Luigi will automatically submit your
calculations to the FireWorks Launchpad. Follow the example in `scripts/update_db.sh`.
the bash_scripts/folder.

And while we're on the subject:  We recommend running `scripts/update_db.sh` on a cron to keep the
databases up-to-date.

## Installation
You will need to add this repo your python path:
```
export PYTHONPATH="/path/to/GASpy:${PYTHONPATH}"
```
This will let GASpy read the `.gaspyrc.json` and import the module from anywhere. Speaking of
which:  You will need to create a `.gaspyrc.json` file and put it in the root directory of
this repository. A template, `.gaspyrc_template.json`, is included. You'll need to set up
FireWorks, a couple of Mongo databases, and Luigi. The consequential login information will be
stored in the `gaspyrc.json`.

Our scripts also assume that you have the `GASpy/` repository either in your home directory or
symlinked to it. If you want to symlink it, then do a:
```
cd
ln -s /path/to/GASpy GASpy/
```

# Under the hood
Since we are performing a (very) large number of DFT simulations, we need to store the data
somewhere. Our Primary, "blessed" database is a MongoDB that is managed by our FireWorks
launchpad.

But our Primary, FireWorks-mananged database is not friendly with our Luigi dependency-mananger.
It is also not able to hold any post-processing information that we want to create (e.g.,
coordination numbers). So we create an Auxiliary vasp.mongo database where we dump all of our
data from our Primary database, but in a "good" format. GASpy then does its administrative work on
the Auxiliary vasp.mongo database and then submits jobs to the Primary FireWorks database. The
user specifies when to update the Auxiliary database from the Primary database (via
`udpate_db.sh`).
