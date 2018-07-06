import os
from collections import OrderedDict
import uuid
import getpass
import numpy as np
from fireworks import Firework, PyTask, LaunchPad, FileWriteTask
import ase.io
from .utils import vasp_settings_to_str, print_dict, read_rc
from . import vasp_functions, defaults
from .vasp_functions import atoms_to_hex


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
                name['name.%s.%s' % (key, key2)] = name_dict[key][key2]
        else:
            if key == 'shift':
                # Search for a range of shift parameters up to 4 decimal place
                shift = float(np.round(name_dict[key], 4))
                name['name.%s' % key] = {'$gte': shift-1e-4, '$lte': shift+1e-4}
            else:
                name['name.%s' % key] = name_dict[key]

    # Get all of the fireworks that are completed, running, or ready (i.e., not fizzled
    # or defused.)
    fw_ids = launchpad.get_fw_ids(name)
    fw_list = []
    for fwid in fw_ids:
        fw = launchpad.get_fw_by_id(fwid)
        if fw.state in ['RUNNING', 'COMPLETED', 'READY']:
            fw_list.append(fwid)
    # Return the matching fireworks
    return fw_list


def make_firework(atoms, fw_name, vasp_setngs, max_atoms=80, max_miller=2):
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
        print('Not making firework because the number of atoms, %i, exceeds the maximum, %i'
              % (len(atoms), max_atoms))
        print_dict(fw_name, indent=1)
        return
    # Notify the user if they try to create a firework with a high miller index
    if 'miller' in fw_name and (np.max(eval(str(fw_name['miller']))) > max_miller):
        print('Not making firework because the miller index exceeds the maximum of %s'
              % max_miller)
        print_dict(fw_name, indent=1)
        return

    # Generate a string representation that we can pass to the job as input
    atom_hex = atoms_to_hex(atoms)
    # Two steps - write the input file and python script to local directory,
    # then relax that traj file
    vasp_filename = vasp_functions.__file__
    if vasp_filename.split('.')[-1] == 'pyc':
        vasp_filename = vasp_filename[:-3] + 'py'

    with open(vasp_filename) as fhandle:
        vasp_functions_contents = fhandle.read()

    write_python_file = FileWriteTask(files_to_write=[{'filename': 'vasp_functions.py',
                                                       'contents': vasp_functions_contents}])

    write_atoms_file = PyTask(func='vasp_functions.hex_to_file',
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
    ''' This function pulls the information about our FireWorks LaunchPad from the config file '''
    # Pull the information from the .gaspyrc
    configs = read_rc()
    lpad = configs['lpad']
    # Make sure that the port is an integer
    lpad['port'] = int(lpad['port'])

    return LaunchPad(**lpad)


def get_firework_info(fw):
    '''
    Given a Fireworks ID, this function will return the "atoms" [class] and
    "vasp_settings" [str] used to perform the relaxation
    '''
    # Pull the atoms objects from the firework. They are encoded, though
    atoms_hex = fw.launches[-1].action.stored_data['opt_results'][1]
    #find the vasp_functions.hex_to_file task atoms object
    vasp_settings = fw.name['vasp_settings']

    # To decode the atoms objects, we need to write them into files and then load
    # them again. To prevent multiple tasks from writing/reading to the same file,
    # we use uuid to create unique file names to write to/read from.
    fname = str(uuid.uuid4()) + '.traj'
    with open(fname, 'w') as fhandle:
        fhandle.write(atoms_hex.decode('hex'))
    atoms = ase.io.read(fname)
    starting_atoms = ase.io.read(fname, index=0)
    os.remove(fname)    # Clean up

    hex_to_file_tasks = [a for a in fw.spec['_tasks']
                         if a['_fw_name'] == 'PyTask' and
                         a['func'] == 'vasp_functions.hex_to_file']

    if len(hex_to_file_tasks) > 0:
        spec_atoms_hex = hex_to_file_tasks[0]['args'][1]
        # To decode the atoms objects, we need to write them into files and then load
        # them again. To prevent multiple tasks from writing/reading to the same file,
        # we use uuid to create unique file names to write to/read from.
        fname = str(uuid.uuid4()) + '.traj'
        with open(fname, 'w') as fhandle:
            fhandle.write(spec_atoms_hex.decode('hex'))
        spec_atoms = ase.io.read(fname)
        os.remove(fname)    # Clean up

        if len(spec_atoms) != len(starting_atoms):
            raise RuntimeError('Spec atoms does not match starting atoms, investigate FW %d' % fw.fw_id)

        # The relaxation often mangles the tags and constraints due to limitations in
        # in vasp() calculators.  We fix this by using the spec constraints/tags
        atoms.set_tags(spec_atoms.get_tags())
        atoms.set_constraint(spec_atoms.constraints)
        starting_atoms.set_tags(spec_atoms.get_tags())
        starting_atoms.set_constraint(spec_atoms.constraints)

    # Guess the pseudotential version if it's not present
    if 'pp_version' not in vasp_settings:
        if 'arjuna' in fw.launches[-1].fworker.name:
            vasp_settings['pp_version'] = '5.4'
        else:
            vasp_settings['pp_version'] = '5.3.5'
        vasp_settings['pp_guessed'] = True
    if 'gga' not in vasp_settings:
        settings = defaults.exchange_correlationals[vasp_settings['xc']]
        for key in settings:
            vasp_settings[key] = settings[key]
    vasp_settings = vasp_settings_to_str(vasp_settings)

    return atoms, starting_atoms, atoms_hex, vasp_settings


def defuse_lost_runs():
    '''
    Sometimes FireWorks desynchronizes with the job management systems, and runs
    become "lost". This function finds and clears them
    '''
    # Find the lost runs
    lp = get_launchpad()
    lost_launch_ids, lost_fw_ids, inconsistent_fw_ids = lp.detect_lostruns()

    # We reverse the list, because early-on in the list there's some really bad
    # launches that cause this script to hang up. If we run in reverse, then
    # it should be able to get the recent ones.
    # TODO:  Find a better solution
    lost_fw_ids.reverse()

    # Defuse them
    for _id in lost_fw_ids:
        lp.defuse_fw(_id)
