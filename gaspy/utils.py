''' Various functions that may be used across GASpy and its submodules '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import os
import pickle
import uuid
import json
import subprocess
from collections import OrderedDict, Iterable, Mapping
from multiprocess import Pool
import numpy as np
import gc
import tqdm
import ase.io
from ase import Atoms
from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from fireworks.core.launchpad import LaunchPad
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


def read_rc(query=None):
    '''
    This function will pull out keys from the .gaspyrc file for you

    Input:
        query   [Optional] The string indicating the configuration you want.
                If you're looking for nested information, use the syntax
                "foo.bar.key"
    Output:
        rc_contents  A dictionary whose keys are the input keys and whose values
                     are the values that we found in the .gaspyrc file
    '''
    rc_file = _find_rc_file()
    with open(rc_file, 'r') as file_handle:
        rc_contents = json.load(file_handle)

    # Return out the keys you asked for. If the user did not specify the key, then return it all
    if query:
        keys = query.split('.')
        for key in keys:
            try:
                rc_contents = rc_contents[key]
            except KeyError as error:
                raise KeyError('Check the spelling/capitalization of the key/values you are looking for').with_traceback(error.__traceback__)

    return rc_contents


def _find_rc_file():
    '''
    This function will search your PYTHONPATH and look for the location of
    your .gaspyrc.json file.

    Returns:
        rc_file     A string indicating the full path to the first .gaspyrc.json
                    file it finds in your PYTHONPATH.
    '''
    # Pull out the PYTHONPATH environment variable
    # so that we know where to look for the .gaspyrc file
    try:
        python_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        raise EnvironmentError('You do not have the PYTHONPATH environment variable. You need to add GASpy to it')

    # Search our PYTHONPATH one-by-one
    rc_file = '.gaspyrc.json'
    for path in python_paths:
        for root, dirs, files in os.walk(path):
            if rc_file in files:
                rc_file = os.path.join(root, rc_file)
                return rc_file

            # If we can find the .gaspyrc_template.json file but not
            # a .gaspyrc.json file yet, then the user probably
            # has their PYTHONPATH set up correctly, but not their
            # .gaspyrc.json file set up, yet.
            if '.gaspyrc_template.json' in files:
                raise EnvironmentError('You have not yet made an appropriate .gaspyrc.json configuration file yet.')


def print_dict(d, indent=0):
    '''
    This function prings a nested dictionary, but in a prettier format. This is strictly for
    debugging purposes.

    Inputs:
        d       The nested dictionary to print
        indent  How many tabs to start the printing at
    '''
    if isinstance(d, dict):
        for key, value in d.items():
            # If the dictionary key is `spec`, then it's going to print out a bunch of
            # messy looking things we don't care about. So skip it.
            if key != 'spec':
                print('\t' * indent + str(key))
                if isinstance(value, dict) or isinstance(value, list):
                    print_dict(value, indent+1)
                else:
                    print('\t' * (indent+1) + str(value))
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, dict) or isinstance(item, list):
                print_dict(item, indent+1)
            else:
                print('\t' * (indent+1) + str(item))
    else:
        pass


def vasp_settings_to_str(vasp_settings):
    '''
    This function is used in various scripts to convert a dictionary of vasp settings
    into a format that is acceptable by ase-db.

    Input:
        vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein
                                may have a different type depending on the VASP setting.
    Output:
        vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein
                                is either an int, float, boolean, or string.
    '''
    vasp_settings = vasp_settings.copy()

    # For each item in "vasp_settings"...
    for key in vasp_settings:
        # Find anything that's not a string, integer, float, or boolean...
        if not isinstance(vasp_settings[key], (str, int, float, bool)):
            # And turn it into a string
            vasp_settings[key] = str(vasp_settings[key])

    return vasp_settings


def docs_to_pdocs(docs):
    '''
    This function turns a list of mongo documents into "parsed" mongo documents.
    Parsed mongo documents are dictionaries of arrays instead of lists of dictionaries.

    Input:
        docs    A list of mongo documents/dictionaries/jsons/whatever you want to call them.
    Output:
        p_docs  A dictionary whose key are identical to the keys in each mongo document,
                but whose values are numpy arrays
    '''
    p_docs = dict.fromkeys(docs[0].keys())
    for key in p_docs:
        p_docs[key] = [doc[key] if key in doc else None for doc in docs]
    return p_docs


def pdocs_to_docs(p_docs):
    '''
    This function turns a list parsed mongo documents into mongo documents.
    Parsed mongo documents are dictionaries of arrays instead of lists of dictionaries.

    Input:
        p_docs  A dictionary whose key are identical to the keys in each mongo document,
                but whose values are numpy arrays
    Output:
        docs    A list of mongo documents/dictionaries/jsons/whatever you want to call them.
    '''
    docs = []
    for i, _ in enumerate(p_docs.values()[0]):
        doc = {}
        for key in p_docs:
            doc[key] = p_docs[key][i]
        docs.append(doc)
    return p_docs


def ads_dict(adsorbate):
    '''
    This is a helper function to take an adsorbate as a string (e.g. 'CO') and attempt to
    return an atoms object for it, primarily as a way to count the number of constitutent
    atoms in the adsorbate.
    '''
    # Try to create an [atoms class] from the input.
    try:
        atoms = Atoms(adsorbate)
    except ValueError:
        print("Not able to create %s with ase.Atoms. Attempting to look in GASpy's dictionary..."
              % adsorbate)

        # If that doesn't work, then look for the adsorbate in our library of adsorbates
        try:
            from .defaults import adsorbates_dict   # Import locally to avoid import errors
            atoms = adsorbates_dict()[adsorbate]
        except KeyError:
            print('%s is not is GASpy library of adsorbates. You need to add it to the adsorbates_dict function in gaspy.defaults'
                  % adsorbate)

    # Return the atoms
    return atoms


def remove_adsorbate(adslab):
    '''
    This sub-function removes adsorbates from an adslab.

    Input:
        adslab  The ase.Atoms object of the adslab. The adsorbate atom(s) must be tagged
                with non-zero integers, while the slab atoms must be tagged with zeroes.
    Outputs:
        slab                The ase.Atoms object of the bare slab. It will not have any
                            constraints of the input `atoms`.
        binding_positions   A dictionary whose keys are the tags of the adsorbates and whose
                            values are the cartesian coordinates of the binding site.
    '''
    # We need to make a local copy of the atoms so that when we start manipulating it here,
    # the changes do not propagate.
    slab = adslab.copy()
    # ASE doesn't like deleting atoms with constraints on them, so we get rid of them.
    slab.set_constraint()
    # Pull out the tags so we know what to delete
    tags = slab.get_tags()
    # Initialize `binding_positions`. We delete the "zero" tag because that's for slabs.
    binding_positions = dict.fromkeys(tags)
    del binding_positions[0]
    # `atoms_range` is a list of atom indices for `slab`. We initialize it because
    # we need to reverse it before iterating through (to maintain proper indexing).
    atoms_range = list(range(0, len(slab)))
    atoms_range.reverse()

    # Delete the adsorbates while simultaneously storing the position of the binding atom.
    for i in atoms_range:
        if tags[i]:
            # Note that we are continuously overwrite the binding positions. The last
            # entry/write # corresponds to the position of the "first" atom for each adsorbate,
            # thus the requirement described in this function's docstring.
            binding_positions[tags[i]] = slab.get_positions()[i]
            del slab[i]

    return slab, binding_positions


def constrain_slab(atoms, z_cutoff=3., symmetric=False):
    '''
    Define a function, "constrain_slab" to impose slab constraints prior to relaxation.

    Inputs:
        atoms       ASE-atoms class of the adsorbate + slab system. The tags of these
                    atoms must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
        z_cutoff    The threshold to see if slab atoms are in the same plane as the
                    highest atom in the slab
        symmetric   If symmetric is False, we constrain everything but the top surface
                    layer. This is the default for adsorption calculations. For surface
                    energy calculations, we want the top and bottom to be relaxed.
    '''
    #copy the input so we don't affect the original (reference) atoms
    atoms = atoms.copy()

    # Remove the adsorbate so that we can find the number of slab atoms
    slab, binding_positions = remove_adsorbate(atoms)
    n_slab_atoms = len(slab)

    # Find the scaled height of the highest and lowest slab atoms.
    scaled_positions = atoms.get_scaled_positions()
    # In an old version of GASpy, we added the adsorbate to the slab instead of adding
    # the slab to the adsorbate. These conditionals determine if the adslab is old or
    # not and processes it accordingly.
    if atoms[0].tag:
        z_max = np.max([pos[2] for pos in scaled_positions[-n_slab_atoms:]])
        z_min = np.min([pos[2] for pos in scaled_positions[-n_slab_atoms:]])
    else:
        z_max = np.max([pos[2] for pos in scaled_positions[:n_slab_atoms]])
        z_min = np.min([pos[2] for pos in scaled_positions[:n_slab_atoms]])

    # Use the scaled heights to fix any atoms below the surfaces of the slab
    constraints = atoms.constraints
    if symmetric:
        constraints.append(FixAtoms(mask=[pos[2] < z_max-(z_cutoff/np.linalg.norm(atoms.cell[2])) and
                                          pos[2] > z_min+(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    elif atoms.cell[2, 2] > 0:
        constraints.append(FixAtoms(mask=[pos[2] < z_max-(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    else:
        constraints.append(FixAtoms(mask=[pos[2] > z_min+(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    atoms.set_constraint(constraints)

    return atoms


def fingerprint_atoms(atoms):
    '''
    This function is used to fingerprint an adslabs atoms object, where the "fingerprint" is a
    dictionary of properties that we believe may be adsorption motifs.

    Inputs:
        atoms   Atoms object to fingerprint. The slab atoms must be tagged with 0 and
                adsorbate atoms must be tagged with non-zero integers. This function also
                assumes that the first atom in each adsorbate is the binding atom (e.g.,
                of all atoms with tag==1, the first atom is the binding; the same goes for
                tag==2 and tag==3 etc.).
    '''
    # Remove the adsorbate(s) while finding the binding position(s)
    atoms, binding_positions = remove_adsorbate(atoms)
    # Add Uranium atoms at each of the binding sites so that we can use them for fingerprinting.
    for tag in binding_positions:
        atoms += Atoms('U', positions=[binding_positions[tag]])

    # Turn the atoms into a pymatgen structure object so that we can use the VCF to find
    # the coordinated sites.
    struct = AseAtomsAdaptor.get_structure(atoms)

    #Test to see if the central atom is entirely on it's own, if so it is not coordinated, so skip the voronoi bit
    # which would throw a QHULL error
    num_cutoff_neighbors = [site[0] for site in enumerate(struct) if 0.1 < struct[len(atoms)-1].distance(site[1]) < 7.0]
    if len(num_cutoff_neighbors) == 0:
        return {'coordination': '',
                'neighborcoord': '',
                'natoms': len(atoms),
                'nextnearestcoordination': ''}

    vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)
    vnn_loose = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)
    try:
        coordinated_atoms_data = vnn.get_nn_info(struct, len(atoms)-1)
    except ValueError:
        vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=40)
        vnn_loose = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=40)
        coordinated_atoms_data = vnn.get_nn_info(struct, len(atoms)-1)
    coordinated_atoms = [atom_data['site'] for atom_data in coordinated_atoms_data]
    # Create a list of symbols of the coordinations, remove uranium from the list, and
    # then turn the list into a single, human-readable string.
    coordinated_symbols = map(lambda x: x.species_string, coordinated_atoms)
    coordinated_symbols = [a for a in coordinated_symbols if a not in ['U']]
    coordination = '-'.join(sorted(coordinated_symbols))

    # Make a [list] of human-readable coordination sites [unicode] for all of the slab atoms
    # that are coordinated to the adsorbate, "neighborcoord"
    neighborcoord = []
    for i in coordinated_atoms:
        # [int] that yields the slab+ads system's atomic index for the 1st-tier-coordinated atom
        neighborind = [site[0] for site in enumerate(struct) if i.distance(site[1]) < 0.1][0]
        # [list] of PyMatGen [periodic site class]es for each of the atoms that are coordinated
        # with the adsorbate
        coord_data = vnn_loose.get_nn_info(struct, neighborind)
        coord = [atom_data['site'] for atom_data in coord_data]
        # The elemental symbols for all of the 2nd-tier-coordinated atoms in a [list] of
        # [unicode] objects
        coord_symbols = map(lambda x: x.species_string, coord)
        # Take out atoms that we assume are not part of the slab
        coord_symbols = [a for a in coord_symbols if a not in ['U']]
        # Sort the list of 2nd-tier-coordinated atoms to induce consistency
        coord_symbols.sort()
        # Turn the [list] of [unicode] values into a single [unicode]
        neighborcoord.append(i.species_string+':'+'-'.join(coord_symbols))

    # [list] of PyMatGen [periodic site class]es for each of the atoms that are
    # coordinated with the adsorbate
    coordinated_atoms_nextnearest_data = vnn_loose.get_nn_info(struct, len(atoms)-1)
    coordinated_atoms_nextnearest = [atom_data['site'] for atom_data in coordinated_atoms_nextnearest_data]
    # The elemental symbols for all of the coordinated atoms in a [list] of [unicode] objects
    coordinated_symbols_nextnearest = map(lambda x: x.species_string,
                                          coordinated_atoms_nextnearest)
    # Take out atoms that we assume are not part of the slab
    coordinated_symbols_nextnearest = [a for a in coordinated_symbols_nextnearest
                                       if a not in ['U']]
    # Turn the [list] of [unicode] values into a single [unicode]
    coordination_nextnearest = '-'.join(sorted(coordinated_symbols_nextnearest))

    # Return a dictionary with each of the fingerprints. Any key/value pair can be added here
    # and will propagate up the chain
    return {'coordination': coordination,
            'neighborcoord': neighborcoord,
            'natoms': len(atoms),
            'nextnearestcoordination': coordination_nextnearest}


def find_adsorption_sites(atoms):
    '''
    A wrapper for pymatgen to get all of the adsorption sites of a slab.

    Arg:
        atoms   The slab where you are trying to find adsorption sites in ase.Atoms format
    Output:
        sites   A list of [array]s, which contain the x-y-z coordinates of the adsorptions sites.
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True)
    sites = sites_dict['all']
    return sites


def find_max_movement(atoms_initial, atoms_final):
    '''
    Given ase.Atoms objects, find the furthest distance that any single atom in a set of
    atoms traveled (in Angstroms)

    Inputs:
        initial_atoms   ase.Atoms objects in their initial state
        final_atoms     ase.Atoms objects in their final state
    '''
    # Calculate the distances for each atom
    distances = atoms_final.positions - atoms_initial.positions

    # Reduce the distances in case atoms wrapped around (the minimum image convention)
    dist, Dlen = find_mic(distances, atoms_final.cell, atoms_final.pbc)

    return np.max(Dlen)


def multimap(function, inputs, chunked=False, processes=32,
             maxtasksperchild=1, chunksize=1, n_calcs=None):
    '''
    This function is a wrapper to parallelize a function.

    Inputs:
        function        The function you want to execute
        inputs          An iterable that yields proper arguments to the function
        chunked         A boolean indicating whether your function expects single arguments
                        or "chunked" iterables.
        processes       The number of threads/processes you want to be using
        maxtaskperchild The maximum number of tasks that a child process may do before
                        terminating (and therefore clearing its memory cache to avoid
                        memory overload).
        chunksize       How many calculations you want to have each single processor do
                        per task. Smaller chunks means more memory shuffling.
                        Bigger chunks means more RAM requirements.
        n_calcs         How many calculations you have. Only necessary for adding a
                        percentage timer to the progress bar.
    Outputs:
        outputs     A list of the inputs mapped through the function
    '''
    # Collect garbage before we begin multiprocessing to make sure we don't pass things we don't
    # need to
    gc.collect()

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        # Use multiprocessing to perform the calculations. We use imap instead of map so that
        # we get an iterator, which we need for tqdm (the progress bar) to work.
        # imap also requires less disk memory, which can be an issue for some of our large
        # systems.
        if not chunked:
            iterator = pool.imap(function, inputs, chunksize=chunksize)
        # If our function expects chunks, then we have to unpack our inputs appropriately
        else:
            iterator = pool.imap(function, (list(arg) for arg in inputs), chunksize=chunksize)
        outputs = list(tqdm.tqdm(iterator, total=n_calcs))

    return outputs


def map_method(instance, method, inputs, chunked=False, processes=32,
               maxtasksperchild=1, chunksize=1, n_calcs=None, **kwargs):
    '''
    This function pools and maps methods of class instances. It does so by putting
    the class instance into the global space so that each worker can pull it.
    This prevents the multiprocessor from pickling/depickling the instance for
    each worker, thus saving time. This is especially needed for lazy learning models
    like GP.

    Note that we set the pickling recursion option to `False` to prevent passing along
    huge instances of objects, which can bottleneck multiprocessing.

    Inputs:
        instance        An instance of a class
        method          A string indicating the method of the class that you want to map
        inputs          An iterable that yields proper arguments to the method
        chunked         A boolean indicating whether your function expects single arguments
                        or "chunked" iterables.
        processes       The number of threads/processes you want to be using
        maxtaskperchild The maximum number of tasks that a child process may do before
                        terminating (and therefore clearing its memory cache to avoid
                        memory overload).
        chunksize       How many calculations you want to have each single processor do
                        per task. Smaller chunks means more memory shuffling.
                        Bigger chunks means more RAM requirements.
        n_calcs         How many calculations you have. Only necessary for adding a
                        percentage timer to the progress bar.
        kwargs          Any arguments that you should be passing to the method
    Outputs:
        outputs     A list of the inputs mapped through the function
    '''
    # Push the class instance to global space for process sharing.
    global module_instance
    module_instance = instance

    # Full the method out of the class instance
    def function(arg):  # noqa: E306
        return getattr(module_instance, method)(arg, **kwargs)

    # Call the `multimap` function for the rest of the work
    outputs = multimap(function, inputs, chunked=chunked,
                       processes=processes,
                       maxtasksperchild=maxtasksperchild,
                       chunksize=chunksize,
                       n_calcs=n_calcs)

    # Clean up and output
    del globals()['module_instance']
    return outputs


def unfreeze_dict(frozen_dict):
    '''
    Recursive function to turn a Luigi frozen dictionary into an ordered dictionary,
    along with all of the branches.

    Arg:
        frozen_dict     Instance of a luigi.parameter._FrozenOrderedDict
    Returns:
        dict_   Ordered dictionary
    '''
    # If the argument is a dictionary, then unfreeze it
    if isinstance(frozen_dict, Mapping):
        unfrozen_dict = OrderedDict(frozen_dict)

        # Recur
        for key, value in unfrozen_dict.items():
            unfrozen_dict[key] = unfreeze_dict(value)

    # Recur on the object if it's a tuple
    elif isinstance(frozen_dict, tuple):
        unfrozen_dict = tuple(unfreeze_dict(element) for element in frozen_dict)

    # Recur on the object if it's a mutable iterable
    elif isinstance(frozen_dict, Iterable) and not isinstance(frozen_dict, str):
        unfrozen_dict = frozen_dict
        for i, element in enumerate(unfrozen_dict):
            unfrozen_dict[i] = unfreeze_dict(element)

    # If the argument is neither mappable nor iterable, we'rpe probably at a leaf
    else:
        unfrozen_dict = frozen_dict

    return unfrozen_dict


def encode_atoms_to_hex(atoms):
    '''
    Encode an atoms object into a hex string.
    Useful when trying to store atoms objects into jsons.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in GASdb (AKA AuxDB), *not* the FireWorks DB.
    '''
    atoms_bytes = pickle.dumps(atoms)
    atoms_hex = atoms_bytes.hex()
    return atoms_hex


def decode_hex_to_atoms(atoms_hex):
    '''
    Decode a hex string into an atoms object.
    Useful when trying to read atoms objects from jsons.

    As of the writing of this docstring, we intend to use this mainly
    to store atoms objects in GASdb (AKA AuxDB), *not* the FireWorks DB.
    '''
    atoms_bytes = bytes.fromhex(atoms_hex)
    atoms = pickle.loads(atoms_bytes, encoding='latin-1')
    return atoms


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


def save_luigi_task_run_results(task, output):
    '''
    This function is a light wrapper to save a luigi task's output. Instead of
    writing the output directly onto the output file, we write onto a temporary
    file and then atomically move the temporary file onto the output file.

    This defends against situations where we may have accidentally queued
    multiple instances of a task; if this happens and both tasks try to write
    to the same file, then the file gets corrupted. But if both of these tasks
    simply write to separate files and then each perform an atomic move, then
    the final output file remains uncorrupted.

    Doing this for more or less every single task in GASpy gots annoying, so
    we wrapped it.

    Args:
        task    Instance of a luigi task whose output you want to write to
        output  Whatever object that you want to save
    '''
    with task.output().temporary_path() as task.temp_output_path:
        with open(task.temp_output_path, 'wb') as file_handle:
            pickle.dump(output, file_handle)


def evaluate_luigi_task(task, force=False):
    '''
    This follows luigi logic to evaluate a task by recursively evaluating all
    requirements. This is useful for executing tasks that are typically
    independent of other tasks, e.g., populating a catalog of sites.

    Arg:
        task    Class instance of a luigi task
        force   A boolean indicating whether or not you want to forcibly
                evaluate the task and all the upstream requirements.
                Useful for re-doing tasks that you know have already been
                completed.
    '''
    # Don't do anything if it's already done and we're not redoing
    if task.complete() and not(force):
        return

    else:
        # Execute prerequisite task[s] recursively
        requirements = task.requires()
        if requirements:
            if isinstance(requirements, Iterable):
                for req in requirements:
                    if not(req.complete()) or force:
                        evaluate_luigi_task(req, force)
            else:
                if not(requirements.complete()) or force:
                    evaluate_luigi_task(requirements, force)

        # Luigi will yell at us if we try to overwrite output files.
        # So if we're foricbly redoing tasks, we need to delete the old outputs.
        if force:
            os.remove(task.output().fn)

        # After prerequisites are done, run the task
        task.run()


def get_lpad():
    '''
    Gets the FireWorks launchpad object according to the login
    information contained in the .gaspyrc.json file.
    '''
    lpad_info = read_rc('lpad')
    lpad_info['port'] = int(lpad_info['port'])
    lpad = LaunchPad(**lpad_info)
    return lpad


def get_final_atoms_object_with_vasp_forces(launch_id):
    # launch_id is the fireworks launch ID
    # returns an atoms object with the correct internal forces
    
    #Set the backup directory
    temp_loc= '/tmp/%s/'%uuid.uuid4()
    
    #Make the directory
    subprocess.call('mkdir %s'%temp_loc, shell=True)
    
    #Unzip the launch backup to the temp directory
    subprocess.call('tar -C %s -xf %s'%(temp_loc,read_rc()['launches_backup_directory']+'/'+'%d.tar.gz'%launch_id), shell=True)
    
    #unzip all of the files if necessary (zipped archive from a backup)
    subprocess.call('gunzip -q %s/* > /dev/null'%temp_loc, shell=True)
    
    #Read the atoms object and use the Vasp2 calculator 
    # to load the correct (DFT) forces from the OUTCAR/etc info
    atoms = ase.io.read('%s/slab_relaxed.traj'%temp_loc)
    vasp2 = Vasp2(atoms, restart=True, directory=temp_loc)
    vasp2.read_results()
    
    #Clean up behind us
    subprocess.call('rm -r %s'%temp_loc, shell=True)
    
    return atoms
