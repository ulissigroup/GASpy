import getpass
import numpy as np
from fireworks import Firework, PyTask
from atoms_to_hex import atoms_to_hex
from print_dict import print_dict


def make_firework(atoms, fw_name, vasp_setngs, max_atoms=50, max_miller=2):
    '''
    This function makes a simple vasp relaxation firework
    atoms       atoms object to relax
    fw_name     dictionary of tags/etc to use as the fireworks name
    vasp_setngs dictionary of vasp settings to pass to Vasp()
    max_atoms   max number of atoms to submit, mainly as a way to prevent overly-large
                simulations from getting run
    max_miller  maximum miller index to submit, so that be default miller indices
                above 3 won't get submitted by accident
    '''
    # Notify the user if they try to create a firework with too many atoms
    if len(atoms) > max_atoms:
        print('        Not making firework because there are too many atoms in the following FW:')
        print_dict(fw_name, indent=3)
        return
    # Notify the user if they try to create a firework with a high miller index
    if 'miller' in fw_name and (np.max(eval(str(fw_name['miller']))) > max_miller):
        print('        Not making firework because the miller index exceeds the maximum of %s' \
              % max_miller)
        print_dict(fw_name, indent=3)
        return

    # Generate a string representation that we can pass to the job as input
    atom_hex = atoms_to_hex(atoms)
    # Two steps - write the input structure to an input file, then relax that traj file
    write_surface = PyTask(func='fireworks_helper_scripts.atoms_hex_to_file',
                           args=['slab_in.traj',
                                 atom_hex])
    opt_bulk = PyTask(func='vasp_scripts.runVasp',
                      args=['slab_in.traj',
                            'slab_relaxed.traj',
                            vasp_setngs],
                      stored_data_varname='opt_results')

    # Package the tasks into a firework, the fireworks into a workflow,
    # and submit the workflow to the launchpad
    fw_name['user'] = getpass.getuser()
    firework = Firework([write_surface, opt_bulk], name=fw_name)
    return firework
