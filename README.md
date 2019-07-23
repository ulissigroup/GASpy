# GASpy:  Generalized Adsorption Simulator for Python

We use Density Functional Theory (DFT) to calculate adsorption energies of adsorbates onto slabs, but we try to do it in a general way such that we may perform these calculations for an arbitrary number of configurations---e.g., different bulk materials, facets, adsorption sites, adsorbates, etc.
We are also able to calculate surface energies and to account for electrolyte, solvent, or voltage effects in electrochemical systems.


# Overview

GASpy is written in [Python 3](https://www.python.org/), and we use various tools (below) that enable us to perform high-throughput DFT relaxations.
You can find a full list of our dependencies in our [Dockerfile](./dockerfile/Dockerfile).

[ASE](https://wiki.fysik.dtu.dk/ase/about.html),
[VASP](https://www.vasp.at/index.php/about-vasp/59-about-vasp),
[Quantum Epsresso](https://www.quantum-espresso.org/),
[PyMatGen](http://pymatgen.org/),
[FireWorks](https://pythonhosted.org/FireWorks/index.html), [Materials Project](https://materialsproject.org/), [Docker](https://www.docker.com/),
[Luigi](https://github.com/spotify/luigi), [MongoDB](https://www.mongodb.com/)

We created various Python classes (referred to as [tasks](https://github.com/ulissigroup/GASpy/tree/master/gaspy/tasks) by [Luigi](https://github.com/spotify/luigi)) to automate adsorption energy calculations.
We use Luigi to manage the dependencies between these tasks, and we use FireWorks to manage/submit our DFT simulations across multiple clusters.
This means that we can simply tell GASpy to "calculate the adsorption energy of CO on Pt", and it automatically performs all of the necessary steps (e.g., fetch Pt from Materials Project; cut slabs; relax the slabs; add CO onto the slab and then relax; then calculate the adsorption energy).

![DAG](./documentation/gaspy_dag.png)

To submit calculations, the simplest thing to do is to use our wrapper functions as such:

    from gaspy.gasdb import get_catalog_docs
    from gaspy.tasks.metadata_calculators import submit_cism_adsorption_calculations
    
    
    # Get all of the sites that we have enumerated
    docs = get_catalog_docs()
    
    # Pick the sites that we want to run. In this case, it'll be sites on
    # palladium (as per Materials Project ID 2, mp-2) on (111) facets.
    site_documents_to_calc = [doc for doc in all_site_documents
                              if (doc['mpid'] == 'mp-2' and
                                  doc['miller'] == [1, 1, 1])]
    
    adsorbate = 'CO'
    submit_adsorption_calculations((adsorbate='CO', catalog_docs=site_documents_to_calc)

This snippet will calculate CO adsorption energies of all sites on the (1, 1, 1) facet of [Pd](https://materialsproject.org/materials/mp-2/).


# Installation & Usage

If you want to get started, please know that GASpy requires a non-trivial amount of overhead setup.
So if you only want to perform a few hundred calculations, we recommend you simply do them manually.
But if you want to run tens of thousands of calculations or more, then GASpy could be for you.
Please refer to our documentation for [installation](./documentation/installation.md) and [usage](./documentation/usage.md) instructions.


# Submodules

You may notice that we have two submodules: [GASpy\_regressions](https://github.com/ulissigroup/GASpy_regressions) and [GASpy\_feedback](https://github.com/ulissigroup/GASpy_feedback).
We use our regression submodule to analyze and perform regressions on our DFT data, and we use our feedback submodule to choose which calculations to \[automatically\] perform next.


# References

[Dynamic Workflows for Routine Materials Discovery in Surface Science](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00386).
We were using GASpy [v0.1](https://github.com/ulissigroup/GASpy/releases/tag/v0.1) for this paper.

[Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution](https://www.nature.com/articles/s41929-018-0142-1).
We were using GASpy [v0.1](https://github.com/ulissigroup/GASpy/releases/tag/v0.1) for this paper.


# Version updates

Current version: 0.30

v0.30:  Added capability to run [Quantum Espresso](https://www.quantum-espresso.org/) and [RISM](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.115429) calculations.

v0.20:  Heavy refactoring and added capability to calculate surface energies.

[v0.10](https://github.com/ulissigroup/GASpy/releases/tag/v0.1):  Initial commits

For an up-to-date list of our software dependencies, you can simply check out how we build our docker image [here](https://github.com/ulissigroup/GASpy/blob/master/dockerfile/Dockerfile).
