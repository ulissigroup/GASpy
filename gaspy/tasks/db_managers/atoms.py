'''
This module houses various functions to update our `atoms` Mongo collection,
which houses the calculation information for each complete calculation we have
in our FireWorks database.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

from datetime import datetime
import warnings
import uuid
import subprocess
import ase
import ase.io
from ase.calculators.vasp import Vasp2
from ... import defaults
from ...utils import read_rc, multimap
from ...mongo import make_doc_from_atoms
from ...gasdb import get_mongo_collection
from ...fireworks_helper_scripts import get_launchpad, get_atoms_from_fw


def update_atoms_collection(n_processes=1):
    '''
    This function will dump all of the completed FireWorks into our `atoms` Mongo
    collection. It will not dump anything that is already there.

    You may notice some functions with the term "patched" in them. If you are
    using GASpy and not in the Ulissigroup, then you can probably skip all the
    patching by not calling the `__patch_old_document` function at all.

    Args:
        n_processes     An integer indicating how many threads you want to use
                        when converting FireWorks into Mongo documents. If you
                        do not expect many updates, stick to the default of 1.
                        If you are re-creating your collection from a full
                        FireWorks database, you may want to increase this
                        argument.
    '''
    fwids_missing = _find_fwids_missing_from_atoms_collection()

    # Make the documents
    print('[%s] Creating %i atoms documents...'
          % (datetime.now(), len(fwids_missing)))
    docs = multimap(_make_atoms_doc_from_fwid, fwids_missing,
                    processes=n_processes, chunksize=100,
                    n_calcs=len(fwids_missing))
    # Clean up `_make_atoms_doc_from_fwid` failures
    docs = [doc for doc in docs if doc is not None]

    # Now write the documents
    if len(docs) > 0:
        print('[%s] Creating %i new entries in the atoms collection...'
              % (datetime.now(), len(docs)))
        with get_mongo_collection('atoms') as collection:
            collection.insert_many(docs)
        print('[%s] Created %i new entries in the atoms collection'
              % (datetime.now(), len(docs)))


def _find_fwids_missing_from_atoms_collection():
    '''
    This method will get the FireWork IDs that are marked as 'COMPLETED' in
    our LaunchPad, but have not yet been added to our `atoms` Mongo
    collection.

    Returns:
        fwids_missing   A set of integers containing the FireWork IDs of
                        the calculations we have not yet added to our
                        `atoms` Mongo collection.
    '''
    with get_mongo_collection('atoms') as collection:
        docs = list(collection.find({}, {'fwid': 'fwid', '_id': 0}))
    fwids_in_atoms = set(doc['fwid'] for doc in docs)

    lpad = get_launchpad()
    fwids_completed = set(lpad.get_fw_ids({'state': 'COMPLETED'}))

    fwids_missing = fwids_completed - fwids_in_atoms
    return fwids_missing


def _make_atoms_doc_from_fwid(fwid):
    '''
    For each fireworks object, turn the results into a mongo doc so that we
    can dump the mongo doc into the Aux DB.

    Args:
        fwid    An integer indicating the FWID you want to turn into a
                document
    Returns:
        doc     A dictionary that contains various information about
                a calculation. Intended to be inserted into Mongo.
    '''
    # Get the `ase.Atoms` objects of the initial and final images
    lpad = get_launchpad()
    fw = lpad.get_fw_by_id(fwid)
    try:
        starting_atoms = get_atoms_from_fw(fw, index=0)
        atoms = get_atoms_from_fw(fw, index=-1)

    # Sometimes the length of the initial atoms and the final atoms are
    # different. If this happens, then defuse the Firework
    except ValueError as error:
        fwid = fw.fw_id
        lpad.defuse_fw(fwid)
        warnings.warn('Defused FireWork %i because the number of initial '
                      'and final atoms differed.' % fwid)
        return None

    # Turn the atoms objects into a document and then add additional
    # information
    doc = make_doc_from_atoms(atoms)
    doc['initial_configuration'] = make_doc_from_atoms(starting_atoms)
    doc['fwname'] = fw.name
    doc['fwid'] = fwid
    doc['directory'] = fw.launches[-1].launch_dir
    doc['calculation_date'] = fw.updated_on

    # Fix some of our old FireWorks
    doc = __patch_old_document(doc, atoms, fw)
    return doc


def __patch_old_document(doc, atoms, fw):
    '''
    We've tried, tested, and failed a lot of times when making FireWorks.
    This method will parse some of our old iterations of FireWorks to meet
    the appropriate standards.

    Arg:
        doc     Dictionary that you'll be adding as a Mongo document to the
                `atoms` collection.
        atoms   Instance of the final `ase.Atoms` image of the Firework you
                are patching. This should NOT be the initial image.
        fw      Instance of the `fireworks.Firework` that you want to patch
    Return:
        patched_doc     Patched version of the `doc` argument.
    '''
    patched_doc = doc.copy()

    # Fix the energies in the atoms object
    patched_atoms = __patch_atoms_from_old_vasp(atoms, fw)
    for key, value in make_doc_from_atoms(patched_atoms).items():
        patched_doc[key] = value

    # Guess some VASP setttings we never recorded
    patched_doc['fwname']['vasp_settings'] = __get_patched_vasp_settings(fw)

    # Some of our old FireWorks had string-formatted Miller indices. Let's
    # fix those.
    if 'miller' in fw.name:
        patched_doc['fwname']['miller'] = __get_patched_miller(fw.name['miller'])

    return patched_doc


def __patch_atoms_from_old_vasp(atoms, fw):
    '''
    The VASP calculator, when used with ASE optimization, was incorrectly
    recording the internal forces in atoms objects with the stored forces
    including constraints. If such incompatible constraints exist and the
    calculations occured before the switch to the Vasp2 calculator, we
    should get the correct (VASP) forces from a backup of the directory
    which includes the INCAR, ase-sort.dat, etc files

    Args:
        atoms   Instance of an `ase.Atoms` object
        fw      Instance of an `fireworks.Firework` object created by the
                `gaspy.fireworks_helper_scripts.make_firework` function
    Returns:
        patched_atoms    Foo
    '''
    # If the FireWork is old and has constraints more complicated than
    # 'FixAtoms', then replace the atoms with another copy that has the
    # correct VASP forces
    if any(constraint.todict()['name'] != 'FixAtoms' for constraint in atoms.constraints):
        if fw.created_on < datetime(2018, 12, 1):
            launch_id = fw.launches[-1].launch_id
            patched_atoms = __get_final_atoms_object_with_vasp_forces(launch_id)
            return patched_atoms

    # If the FireWork is new or simple, then no patching is necessary
    return atoms


def __get_final_atoms_object_with_vasp_forces(launch_id):
    '''
    This function will return an ase.Atoms object from a particular
    FireWorks launch ID. It will also make sure that the ase.Atoms object
    will have VASP-calculated forces attached to it.

    Arg:
        launch_id   An integer representing the FireWorks launch ID of the
                    atoms object you want to get
    Returns:
        atoms   ase.Atoms object with the VASP-calculated forces
    '''
    # We will be opening a temporary directory where we will unzip the
    # FireWorks launch directory
    fw_launch_file = (read_rc('fireworks_info.backup_directory') +
                      '/%d.tar.gz' % launch_id)
    temp_loc = __dump_file_to_tmp(fw_launch_file)

    # Load the atoms object and then load the correct (DFT) forces from the
    # OUTCAR/etc info
    try:
        atoms = ase.io.read('%s/slab_relaxed.traj' % temp_loc)
        vasp2 = Vasp2(atoms, restart=True, directory=temp_loc)
        vasp2.read_results()

    # Clean up behind us
    finally:
        subprocess.call('rm -r %s' % temp_loc, shell=True)

    return atoms


def __dump_file_to_tmp(file_name):
    '''
    Take the contents of a directory and then dump it into a temporary
    directory while simultaneously unzipping it. This makes reading from
    FireWorks directories faster.

    Arg:
        file_name   String indicating what directory you want to dump
    Returns:
        temp_loc    A string indicating where we just dumped the directory
    '''
    # Make the temporary directory
    temp_loc = '/tmp/%s/' % uuid.uuid4()
    subprocess.call('mkdir %s' % temp_loc, shell=True)

    # Move to the temporary folder and unzip everything
    subprocess.call('tar -C %s -xf %s' % (temp_loc, file_name), shell=True)
    subprocess.call('gunzip -q %s/* > /dev/null' % temp_loc, shell=True)

    return temp_loc


def __get_patched_vasp_settings(fw):
    '''
    Gets the VASP settings we used for a FireWork. Some of them are old and
    have missing information, so this function will also fill in some
    blanks.  If you're using GASpy outside of the Ulissi group, then you
    should probably delete all of the lines past
    `vasp_settings=fw.name['vasp_settings']`.

    Arg:
        fw  Instance of a `fireworks.core.firework.Firework` class that
            should probably get obtained from our Launchpad
    Returts:
        vasp_settings   A dictionary containing the VASP settings we used
                        to perform the relaxation for this FireWork.
    '''
    vasp_settings = fw.name['vasp_settings']

    # Guess the pseudotential version if it's not present
    if 'pp_version' not in vasp_settings:
        if 'arjuna' in fw.launches[-1].fworker.name:
            vasp_settings['pp_version'] = '5.4'
        else:
            vasp_settings['pp_version'] = '5.3.5'
        vasp_settings['pp_guessed'] = True

    # ...I think this is to patch some of our old, incorrectly tagged
    # calculations? I'm not sure. I think that it just assumes that
    # untagged calculations are RPBE.
    if 'gga' not in vasp_settings:
        settings = defaults.xc_settings(xc='rpbe')
        for key in settings:
            vasp_settings[key] = settings[key]

    return vasp_settings


def __get_patched_miller(miller):
    '''
    Some of our old Fireworks have Miller indices stored as strings instead
    of arrays of integers. We fix that here.

    Arg:
        miller          Either a string or a sequence indicating the Miller
                        indices
    Returns:
        patched_miller  The Miller indices formatted as a tuple of integers
    '''
    if isinstance(miller, str):     # Only patch it if the format is incorrect
        patched_miller = tuple([int(index) for index in miller.lstrip('(').rstrip(')').split(',')])
        return patched_miller
    else:
        return miller
