from collections import OrderedDict
import uuid
import getpass
import numpy as np
from fireworks import Firework, PyTask, LaunchPad, FileWriteTask
import ase.io
from vasp import Vasp
from utils import vasp_settings_to_str, print_dict
import vasp_functions
from vasp_functions import atoms_to_hex, hex_to_file


def running_fireworks(name_dict, launchpad):
    '''
    Return the running, ready, or completed fireworks on the launchpad with a given name
    name_dict   name dictionary to search for
    launchpad   launchpad to use
    '''
    # Make a mongo query
    name = {}
    # Turn a nested dictionary into a series of mongo queries
    for key in name_dict:
        if isinstance(name_dict[key], dict) or isinstance(name_dict[key], OrderedDict):
            for key2 in name_dict[key]:
                name['name.%s.%s'%(key, key2)] = name_dict[key][key2]
        else:
            if key == 'shift':
                # Search for a range of shift parameters up to 4 decimal place
                shift = float(np.round(name_dict[key], 4))
                name['name.%s'%key] = {'$gte':shift-1e-4, '$lte':shift+1e-4}
            else:
                name['name.%s'%key] = name_dict[key]

    # Get all of the fireworks that are completed, running, or ready (i.e., not fizzled
    # or defused.)
    fw_ids = launchpad.get_fw_ids(name)
    fw_list = []
    for fwid in fw_ids:
        fw = launchpad.get_fw_by_id(fwid)
        if fw.state in ['RUNNING', 'COMPLETED', 'READY']:
            fw_list.append(fwid)
    # Return the matching fireworks
    if len(fw_list) == 0:
        print('        No matching FW for:')
        print_dict(name, indent=3)
    return fw_list


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
    # Two steps - write the input file and python script to local directory,
    # then relax that traj file

    with open(vasp_functions.__file__) as fhandle:
        vasp_functions_contents=fhandle.read()

    write_python_file = FileWriteTask(files_to_write=[{'filename':'vasp_functions.py',
                                                       'contents': vasp_functions_contents}])

    write_atoms_file =  PyTask(func='vasp_functions.hex_to_file',
                               args=['slab_in.traj', atom_hex])

    opt_bulk = PyTask(func='vasp_functions.runVasp',
                      args=['slab_in.traj', 'slab_relaxed.traj', vasp_setngs],
                      stored_data_varname='opt_results')

    # Package the tasks into a firework, the fireworks into a workflow,
    # and submit the workflow to the launchpad
    fw_name['user'] = getpass.getuser()
    firework = Firework([write_python_file, write_atoms_file, opt_bulk], name=fw_name)
    return firework


def get_launchpad():
    ''' This function contains the information about our FireWorks LaunchPad '''
    return LaunchPad(host='mongodb01.nersc.gov',
                     name='fw_zu_vaspsurfaces',
                     username='admin_zu_vaspsurfaces',
                     password='$TPAHPmj',
                     port=27017)


def get_firework_info(fw):
    '''
    Given a Fireworks ID, this function will return the "atoms" [class] and
    "vasp_settings" [str] used to perform the relaxation
    '''
    # Pull the atoms objects from the firework. They are encoded, though
    atoms_hex = fw.launches[-1].action.stored_data['opt_results'][1]
    vasp_settings = fw.name['vasp_settings']

    # To decode the atoms objects, we need to write them into files and then load
    # them again. To prevent multiple tasks from writing/reading to the same file,
    # we use uuid to create unique file names to write to/read from.
    with stri(uuid.uuid4()) + '.traj' as fname:
        with open(fname, 'w') as fhandle:
            fhandle.write(atoms_hex.decode('hex'))
            atoms = ase.io.read(fname)
            starting_atoms = ase.io.read(fname, index=0)
            os.remove(fname)    # Clean up

    # Guess the pseudotential version if it's not present
    if 'pp_version' not in vasp_settings:
        if 'arjuna' in fw.launches[-1].fworker.name:
            vasp_settings['pp_version'] = '5.4'
        else:
            vasp_settings['pp_version'] = '5.3.5'
        vasp_settings['pp_guessed'] = True
    if 'gga' not in vasp_settings:
        settings = Vasp.xc_defaults[vasp_settings['xc']]
        for key in settings:
            vasp_settings[key] = settings[key]
    vasp_settings = vasp_settings_to_str(vasp_settings)

    return atoms, starting_atoms, atomshex, vasp_settings
