'''
This submodule contains various functions that help us manage
and interact with FireWorks.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import os
import warnings
import uuid
import getpass
import pandas as pd
import ase.io
from fireworks import Firework, PyTask, LaunchPad, FileWriteTask, Workflow
from .utils import vasp_settings_to_str, print_dict, read_rc
from . import vasp_functions, defaults


def get_launchpad():
    '''
    This function returns an instance of a `fireworks.LaunchPad` object that is
    connected to our FireWorks launchpad.

    Returns:
        lpad    An instance of a `fireworks.LaunchPad` object
    '''
    configs = read_rc('fireworks_info.lpad')
    configs['port'] = int(configs['port'])  # Make sure that the port is an integer
    lpad = LaunchPad(**configs)
    return lpad


def is_rocket_running(query, vasp_settings, _testing=False):
    '''
    This function will check if we have something currently running in our
    FireWorks launcher. It will also warn you if we have a lot of fizzles.

    Args:
        query           A dictionary that can be passed as a `query` argument
                        to the `fireworks` collection of our FireWorks database.
        vasp_settings   A dictionary of vasp settings. These will be
                        automatically parsed into the `query` argument.
        _testing        Boolean indicating whether or not you are currently
                        doing a unit test. You should probably not be
                        changing the default from False.
    Returns:
        A boolean indicating whether or not we are currently running any
        FireWorks rockets that match the query and VASP settings
    '''
    # Parse the VASP settings into the FireWorks query, then grab the docs
    # and warn the user if there are a bunch of fizzles
    for key, value in vasp_settings.items():
        query['name.vasp_settings.%s' % key] = value
    docs = _get_firework_docs(query=query, _testing=_testing)
    __warn_about_fizzles(docs)

    # Check if we are running currently. 'COMPLETED' is considered running
    # because it's offically done when it's in our atoms collection, not
    # when it's done in FireWorks
    running_states = set(['COMPLETED', 'READY', 'RESERVED', 'RUNNING', 'PAUSED'])
    docs_running = [doc for doc in docs if doc['state'] in running_states]
    if len(docs_running) > 0:
        print('You just asked if the following FireWork rocket is running. '
              'We have %i of those running. The FWID[s] is/are (%s) in (%s) '
              'states.' % (len(docs_running),
                           ', '.join(str(doc['fw_id']) for doc in docs_running),
                           ', '.join(doc['state'] for doc in docs_running)))
        print_dict(query, indent=1)
        return True
    else:
        return False


def _get_firework_docs(query, _testing):
    '''
    This function will get some documents from our FireWorks database.

    Args:
        query       A dictionary that can be passed as a `query` argument
                    to the `fireworks` collection of our FireWorks database.
        _testing    Boolean indicating whether or not you are currently
                    doing a unit test. You should probably not be
                    changing the default from False.
    Returns:
        docs    A list of dictionaries (i.e, Mongo documents) obtained
                from the `fireworks` collection of our FireWorks Mongo.
    '''
    lpad = get_launchpad()

    # Grab the correct collection, depending on whether or not we are
    # unit testing
    if _testing is False:
        collection = lpad.fireworks
    else:
        collection = lpad.fireworks.database.get_collection('unit_testing_fireworks')

    try:
        docs = list(collection.find(query))
    finally:    # Make sure we close the connection
        collection.database.client.close()
    return docs


def __warn_about_fizzles(docs):
    '''
    If we've tried a bunch of times before and kept failing, then let the user
    know.
    '''
    docs_fizzled = [doc for doc in docs if doc['state'] == 'FIZZLED']
    fwids_fizzled = [str(doc['fw_id']) for doc in docs_fizzled]
    if len(docs_fizzled) > 0:
        message = ('We have fizzled a calculation %i time[s] so far. '
                   'The FireWork IDs are:  %s'
                   % (len(docs_fizzled), ', '.join(fwids_fizzled)))
        warnings.warn(message, RuntimeWarning)


def make_firework(atoms, fw_name, vasp_settings):
    '''
    This function makes a FireWorks rocket to perform a VASP relaxation

    Args:
        atoms           `ase.Atoms` object to relax
        fw_name         Dictionary of tags/etc to use as the FireWorks name
        vasp_settings   Dictionary of VASP settings to pass to Vasp()
    Returns:
        firework    An instance of a `fireworks.Firework` object that is set up
                    to perform a VASP relaxation
    '''
    # Take the `vasp_functions` submodule in GASpy and then pass it out to each
    # FireWork rocket to use.
    vasp_filename = vasp_functions.__file__
    if vasp_filename.split('.')[-1] == 'pyc':   # Make sure we use the source file
        vasp_filename = vasp_filename[:-3] + 'py'
    with open(vasp_filename) as file_handle:
        vasp_functions_contents = file_handle.read()
    pass_vasp_functions = FileWriteTask(files_to_write=[{'filename': 'vasp_functions.py',
                                                         'contents': vasp_functions_contents}])

    # Convert the atoms object to a string so that we can pass it through
    # FireWorks, and then tell the FireWork rocket to use our `vasp_functions`
    # submodule to unpack the string
    atom_trajhex = encode_atoms_to_trajhex(atoms)
    read_atoms_file = PyTask(func='vasp_functions.hex_to_file',
                             args=['slab_in.traj', atom_trajhex])

    # Tell the FireWork rocket to perform the relaxation
    relax = PyTask(func='vasp_functions.runVasp',
                   args=['slab_in.traj', 'slab_relaxed.traj', vasp_settings],
                   stored_data_varname='opt_results')

    fw_name['user'] = getpass.getuser()
    firework = Firework([pass_vasp_functions, read_atoms_file, relax], name=fw_name)
    return firework


def encode_atoms_to_trajhex(atoms):
    '''
    Encode a trajectory-formatted atoms object into a hex string.
    Differs from `encode_atoms_to_hex` since this method is hex-encoding
    the trajectory, not an atoms object.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in the FireWorks DB, *not* the GASdb (AKA AuxDB).

    Arg:
        atoms   ase.Atoms object to encode
    Output:
        hex_    A hex-encoded string object of the trajectory of the atoms object
    '''
    # Make the trajectory
    fname = read_rc('temp_directory') + str(uuid.uuid4()) + '.traj'
    atoms.write(fname)

    # Encode the trajectory
    with open(fname, 'rb') as fhandle:
        hex_ = fhandle.read().hex()

    # Clean up
    os.remove(fname)
    return hex_


def decode_trajhex_to_atoms(hex_, index=-1):
    '''
    Decode a trajectory-formatted atoms object into a hex string.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in the FireWorks DB, *not* the GASdb (AKA AuxDB).

    Arg:
        hex_    A hex-encoded string of a trajectory of atoms objects.
        index   Trajectories can contain multiple atoms objects.
                The `index` is used to specify which atoms object to return.
                -1 corresponds to the last image.
    Output:
        atoms   The decoded ase.Atoms object
    '''
    # Make the trajectory from the hex
    fname = read_rc('temp_directory') + str(uuid.uuid4()) + '.traj'
    with open(fname, 'wb') as fhandle:
        fhandle.write(bytes.fromhex(hex_))

    # Open up the atoms from the trajectory
    atoms = ase.io.read(fname, index=index)

    # Clean up
    os.remove(fname)
    return atoms


def submit_fwork(fwork, _testing=False):
    '''
    This function will package a FireWork into a workflow for you and then add
    it to our FireWorks launchpad.

    Args:
        fwork       Instance of a `fireworks.Firework` object
        _testing    Boolean indicating whether or not you're doing a unit test.
                    You probably shouldn't touch this.
    Returns:
        wflow   An instance of the `firework.Workflow` that was added to the
                FireWorks launch pad.
    '''
    wflow = Workflow([fwork], name='vasp optimization')

    if not _testing:
        lpad = get_launchpad()
        lpad.add_wf(wflow)
        print('Just submitted the following FireWork rocket:')
        print_dict(fwork.name, indent=1)

    return wflow


def get_firework_info(fw):
    '''
    Given a Fireworks ID, this function will return the "atoms" [class] and
    "vasp_settings" [str] used to perform the relaxation
    '''
    # Pull the atoms objects from the firework. They are encoded, though
    atoms_trajhex = fw.launches[-1].action.stored_data['opt_results'][1]
    # find the vasp_functions.hex_to_file task atoms object
    vasp_settings = fw.name['vasp_settings']

    atoms = decode_trajhex_to_atoms(atoms_trajhex)
    starting_atoms = decode_trajhex_to_atoms(atoms_trajhex, index=0)

    hex_to_file_tasks = [a for a in fw.spec['_tasks']
                         if a['_fw_name'] == 'PyTask' and
                         a['func'] == 'vasp_functions.hex_to_file']

    if len(hex_to_file_tasks) > 0:
        spec_atoms_hex = hex_to_file_tasks[0]['args'][1]
        # To decode the atoms objects, we need to write them into files and then load
        # them again. To prevent multiple tasks from writing/reading to the same file,
        # we use uuid to create unique file names to write to/read from.
        spec_atoms = decode_trajhex_to_atoms(spec_atoms_hex)

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
        settings = defaults.exchange_correlational_settings()[vasp_settings['xc']]
        for key in settings:
            vasp_settings[key] = settings[key]
    vasp_settings = vasp_settings_to_str(vasp_settings)

    return atoms, starting_atoms, atoms_trajhex, vasp_settings


def defuse_lost_runs():
    '''
    Sometimes FireWorks desynchronizes with the job management systems, and runs
    become "lost". This function finds and clears them
    '''
    # Find the lost runs
    lpad = get_launchpad()
    lost_launch_ids, lost_fw_ids, inconsistent_fw_ids = lpad.detect_lostruns()

    # We reverse the list, because early-on in the list there's some really bad
    # launches that cause this script to hang up. If we run in reverse, then
    # it should be able to get the recent ones.
    # TODO:  Find a better solution
    lost_fw_ids.reverse()

    # Defuse them
    for _id in lost_fw_ids:
        lpad.defuse_fw(_id)


def check_jobs_status(user_ID, num_jobs):
    '''
    This function returns the status of the submitted FW_jobs as a pandas
    dataframe. The job status are displayed in reversed order (last job to
    first job).

    For example, if Zack submitted 2 jobs (with fwid 10000, 10001),
    and wants to check their status, he will get
    fwi    mpid    miller_index    shift    top    calculation_type             user       job status    directory
    10001  mp-XXX  [1, 1, 0]       0.0      True   slab+adsorbate optimization  'zulissi'  RUNNING       /home/zulissi/fireworks/block_2018-11-25-21-28...
    10000  mp-YYY  [1, 1, 0]       0.0      True   slab+adsorbate optimization  'zulissi'  RUNNING       /home-research/zulissi/fireworks/blocks/block_...

    Args:
        user:       Your cori user ID, which is usually your CMU andrew ID, input as a string.
                    For example: 'zulissi' or 'ktran'.
        num_jobs    Number of submitted job you want to check
    Returns:
        dataframe   A Pandas DataFrame that contains FW job status. Information includes:
                    user, mpid, miller index, shift, calculation_type (e.g slab_adsorbate
                    optimization, slab optimization), top, adsorbate (if any), job status
                    (e.g. COMPLETED, FIZZLED, READY, DEFUSED), and directories.
    '''
    lpad = get_launchpad()
    user_fwids = lpad.get_fw_ids({'name.user': user_ID})
    user_fwids.sort(reverse=True)

    fireworks_info = []
    for fwid in user_fwids[:num_jobs]:
        fw = lpad.get_fw_by_id(fwid)

        # EAFP to get the launch directory, which does not exists for unlaunched fireworks
        try:
            launch_dir = fw.launches[0].launch_dir
        except IndexError:
            launch_dir = ''

        fw_info = (fwid,
                   fw.name.get('mpid', ''),
                   fw.name.get('miller', ''),
                   fw.name.get('shift', ''),
                   fw.name.get('top', ''),
                   fw.name.get('calculation_type', ''),
                   fw.name.get('adsorbate', ''),
                   fw.name.get('user', ''),
                   fw.state,
                   launch_dir)
        fireworks_info.append(fw_info)

    data_labels = ('fwid',
                   'mpid',
                   'miller',
                   'shift',
                   'top',
                   'calculation_type',
                   'adsorbate',
                   'user',
                   'state',
                   'launch_dir')
    dataframe = pd.DataFrame(fireworks_info, columns=data_labels)

    return dataframe


#def running_fireworks(name_dict, launchpad):
#    '''
#    Return the running, ready, or completed fireworks on the launchpad with a given name
#    name_dict   name dictionary to search for
#    launchpad   launchpad to use
#    '''
#    # Make a mongo query
#    name = {}
#    # Turn a nested dictionary into a series of mongo queries
#    for key in name_dict:
#        if isinstance(name_dict[key], dict) or isinstance(name_dict[key], OrderedDict):
#            for key2 in name_dict[key]:
#                name['name.%s.%s' % (key, key2)] = name_dict[key][key2]
#        else:
#            if key == 'shift':
#                # Search for a range of shift parameters up to 4 decimal place
#                shift = float(np.round(name_dict[key], 4))
#                name['name.%s' % key] = {'$gte': shift-1e-4, '$lte': shift+1e-4}
#            else:
#                name['name.%s' % key] = name_dict[key]
#
#    # Get all of the fireworks that are completed, running, or ready (i.e., not fizzled
#    # or defused.)
#    fw_ids = launchpad.get_fw_ids(name)
#    fw_list = []
#    for fwid in fw_ids:
#        fw = launchpad.get_fw_by_id(fwid)
#        if fw.state in ['RUNNING', 'COMPLETED', 'READY']:
#            fw_list.append(fwid)
#    # Return the matching fireworks
#    return fw_list
