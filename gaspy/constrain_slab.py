import numpy as np
from ase.constraints import FixAtoms


def constrain_slab(atoms, n_atoms, z_cutoff=3.):
    '''
    Define a function, "constrain_slab" to impose slab constraints prior to relaxation.
    Inputs
    atoms       ASE-atoms class of the slab to be constrained
    n_atoms     number of slab atoms
    z_cutoff    The threshold to see if other atoms are in the same plane as the highest atom
    '''
    # Initialize
    constraints = []        # This list will contain the various constraints we will impose

    # Constrain atoms except for the top layer. To do this, we first pull some information out
    # of the atoms object.
    scaled_positions = atoms.get_scaled_positions() #
    z_max = np.max([pos[2] for pos in scaled_positions[0:n_atoms]]) # Scaled height of highest atom
    z_min = np.min([pos[2] for pos in scaled_positions[0:n_atoms]]) # Scaled height of lowest atom
    # Add the constraint, which is a binary list (i.e., 1's & 0's) used to identify which atoms
    # to fix or not. The indices of the list correspond to the indices of the atoms in the "atoms".
    if atoms.cell[2, 2] > 0:
        constraints.append(FixAtoms(mask=[pos[2] < z_max-(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    else:
        constraints.append(FixAtoms(mask=[pos[2] > z_min+(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))

    # Enact the constraints on the local atoms class
    atoms.set_constraint(constraints)
    return atoms

#We use this function to determine which side is the "top" side
#def calculate_top(atoms,num_adsorbate_atoms=0):
#    if num_adsorbate_atoms>0:
#        atoms=atoms[0:-num_adsorbate_atoms]
#    zpos=atoms.positions[:,2]
#    return np.sum((zpos-zpos.mean())*atoms.get_masses())>0
