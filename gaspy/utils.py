''' Various functions that may be used across GASpy and its submodules '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb  # noqa: F401
from pprint import pprint
import dill as pickle
import numpy as np
from multiprocess import Pool
import gc
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
import tqdm
from . import defaults, readrc
from luigi.parameter import _FrozenOrderedDict
from luigi.target import FileAlreadyExists
import collections
from collections import OrderedDict
import uuid
import ase.io
import os


def read_rc(key=None):
    '''
    This function will pull out keys from the .gaspyrc file for you.
    Note that this is really only here as a pointer so that the user can look for
    the function in both this module and the native module, `readrc`

    Input:
        keys    [Optional] The string indicating the configuration you want
    Output:
        configs A dictionary whose keys are the input keys and whose values
                are the values that we found in the .gaspyrc file
    '''
    configs = readrc.read_rc(key)
    return configs


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
        pprint("Not able to create %s with ase.Atoms. Attempting to look in GASpy's dictionary..."
               % adsorbate)

        # If that doesn't work, then look for the adsorbate in our library of adsorbates
        try:
            atoms = defaults.adsorbates_dict()[adsorbate]
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
    This function is a wrapper to parallelize a function. Note that we set the pickling
    recursion option to `False` to prevent passing along huge instances of objects,
    which can bottleneck multiprocessing.

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
    pool = Pool(processes=processes, maxtasksperchild=maxtasksperchild)

    # We set pickle recursion to false so that we don't accidentally pass unneccessary information
    # to each thread
    pickle.settings['recurse'] = False

    # Use multiprocessing to perform the calculations. We use imap instead of map so that
    # we get an iterator, which we need for tqdm (the progress bar) to work.
    if not chunked:
        iterator = pool.imap(function, inputs, chunksize=chunksize)
    # If our function expects chunks, then we have to unpack our inputs appropriately
    else:
        iterator = pool.imap(function, (list(arg) for arg in inputs), chunksize=chunksize)
    outputs = list(tqdm.tqdm(iterator, total=n_calcs))

    # Clean up and output
    pool.terminate()
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
    frozen_dict = OrderedDict(frozen_dict)
    for key in frozen_dict:
        if type(frozen_dict[key]) == _FrozenOrderedDict:
            frozen_dict[key] = unfreeze_dict(frozen_dict[key])
    return frozen_dict


def encode_atoms_to_hex(atoms):
    '''
    Encode an atoms object into a hex string. Useful when trying to
    store atoms objects into jsons.
    '''
    atoms_bytes = pickle.dumps(atoms)
    atoms_hex = atoms_bytes.hex()
    return atoms_hex


def decode_hex_to_atoms(atoms_hex):
    '''
    Decode a hex string into an atoms object. Useful when trying to
    read atoms objects from jsons.
    '''
    atoms_bytes = bytes.fromhex(atoms_hex)
    atoms = pickle.loads(atoms_bytes)
    return atoms


def decode_trajhex_to_atoms(trajhex, index=-1):
    fname = str(uuid.uuid4()) + '.traj'
    with open(fname, 'wb') as fhandle:
        fhandle.write(bytes.fromhex(trajhex))
    atoms = ase.io.read(fname, index=index)
    os.remove(fname)
    return atoms


def encode_atoms_to_trajhex(atoms):
    fname = str(uuid.uuid4()) + '.traj'
    atoms.write(fname)
    with open(fname, 'rb') as fhandle:
        hexstr = fhandle.read().hex()
    os.remove(fname)
    return hexstr


def luigi_task_eval(task):
    '''
    This follow luigi logic to evaluate a task by recursively evaluating all requirements.
    This is useful for executing tasks that are typically independennt of other tasks,
    e.g., populating a catalog of sites.

    Arg:
        task    Class instance of a luigi task
    '''
    if task.complete():
        return
    else:
        task_req = task.requires()
        if task_req:
            if isinstance(task_req, collections.Iterable):
                for req in task.requires():
                    if not(req.complete()):
                        luigi_task_eval(req)
            else:
                if not(task_req.complete()):
                    luigi_task_eval(task_req)
        try:
            task.run()
        except FileAlreadyExists:
            return

