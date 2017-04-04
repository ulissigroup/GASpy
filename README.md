# GASpy

Generalized Adsorptions Simulator for Python

This is a collection of scripts to populate an ASE database of DFT calculations of bulk and gas structures from Materials Project.

Luigi is called to manage the dependencies of the different calculations (e.g., need slab relaxations before adsorption relaxations). Fireworks is used to submit the calculations. ASE and PyMatGen are used to manage the DFT objects. VASP is used to perform the relaxations.


