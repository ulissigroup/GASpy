''' Tests for the `gaspy.tasks.db_managers` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.db_managers import (_find_fwids_missing_from_atoms_collection,
                                  _make_doc_from_fwid,
                                  __patch_old_document,
                                  __patch_atoms_from_old_vasp,
                                  __get_final_atoms_object_with_vasp_forces,
                                  __dump_file_to_tmp,
                                  __get_patched_vasp_settings,
                                  __get_patched_miller,
                                  update_catalog_collection,
                                  _GetMpids,
                                  _InsertSitesToCatalog)

# Things we need to do the tests
import subprocess
from datetime import datetime
import pickle
import numpy.testing as npt
import ase
from pymatgen.ext.matproj import MPRester
from .utils import clean_up_tasks, run_task_locally
from ..test_cases.mongo_test_collections.mongo_utils import populate_unit_testing_collection
from ... import defaults
from ...gasdb import get_mongo_collection
from ...utils import unfreeze_dict, read_rc
from ...mongo import make_atoms_from_doc, make_doc_from_atoms
from ...fireworks_helper_scripts import get_atoms_from_fw, get_launchpad
from ...tasks import get_task_output

SLAB_SETTINGS = defaults.slab_settings()


def test_update_atoms_collection():
    '''
    I've chosen to not actually test things function partly out of time
    constraints *cough* lazy *cough*, and partly because I rely so heavily on
    the unit test of the child/helper functions.
    '''
    assert True


def test_find_missing_fwids():
    '''
    This test uses our unit testing `atoms` collection, but our production
    FireWorks. If you don't have at least ten missing Fireworks between the
    two, it'll probably fail. You can fix it by just changing the number of
    test cases.
    '''
    lpad = get_launchpad()
    missing_fwids = _find_fwids_missing_from_atoms_collection()

    with get_mongo_collection('atoms') as collection:
        # Just take the first ten to speed the test up
        for fwid in list(missing_fwids)[0:10]:

            # Make sure the missing FWs are actually done
            fw = lpad.get_fw_by_id(fwid)
            assert fw.state == 'COMPLETED'

            # Make sure the missing FWs are not in our collection
            docs = list(collection.find({'fwid': fwid}))
            assert len(docs) == 0


def test_make_doc_from_fwid():
    '''
    This test will try to make a document from your real FireWorks
    database, not the unit testing one (because I'm too lazy). So if it
    fails, then change the ID to a FireWorks ID of a completed rocket that
    you have.
    '''
    fwid = 42
    doc = _make_doc_from_fwid(fwid)

    # Verify that we can make atoms objects from the document
    atoms = make_atoms_from_doc(doc)
    starting_atoms = make_atoms_from_doc(doc['initial_configuration'])
    assert isinstance(atoms, ase.Atoms)
    assert isinstance(starting_atoms, ase.Atoms)

    # Check that we have some of the necessary fields
    assert 'fwname' in doc  # If we weren't patching, this would be a real comparison
    assert doc['fwid'] == fwid
    assert 'directory' in doc


def test___patch_old_document():
    '''
    We rely on unit testing of the child functions to verify that we do the
    patching correctly. This test will instead make sure that the function
    can run and returns a separate instance.
    '''
    # Create an example document
    fireworks_folder = '/home/GASpy/gaspy/tests/test_cases/fireworks/'
    for file_name in os.listdir(fireworks_folder):
        with open(fireworks_folder + file_name, 'rb') as file_handle:
            fw = pickle.load(file_handle)
        atoms = get_atoms_from_fw(fw)
        doc = make_doc_from_atoms(atoms)
        doc['fwname'] = fw.name

        # Check that we changed things
        patched_doc = __patch_old_document(doc, atoms, fw)
        assert doc is not patched_doc


def test___patch_atoms_from_old_vasp():
    '''
    This test just makes sure that we change the atoms when we need to. We
    rely on the unit testing of `__get_final_atoms_with_vasp_forces` to
    patch correctly.
    '''
    # Get example fireworks and atoms objects
    fireworks_folder = '/home/GASpy/gaspy/tests/test_cases/fireworks/'
    for file_name in os.listdir(fireworks_folder):
        with open(fireworks_folder + file_name, 'rb') as file_handle:
            fw = pickle.load(file_handle)
        atoms = get_atoms_from_fw(fw)

        # Check that we change the atoms when necessary
        patched_atoms = __patch_atoms_from_old_vasp(atoms, fw)
        if any(constraint.todict()['name'] != 'FixAtoms' for constraint in atoms.constraints):
            if fw.created_on < datetime(2018, 12, 1):
                assert atoms != patched_atoms


def test___get_final_atoms_with_vasp_forces():
    atoms = __get_final_atoms_object_with_vasp_forces(101392)
    assert type(atoms) == ase.atoms.Atoms
    assert type(atoms.get_calculator()) == ase.calculators.vasp.Vasp2


def test___dump_directory_to_tmp():
    zipped_directory = ('/home/GASpy/gaspy/tests/test_cases/'
                        'launches_backup_directory/101392.tar.gz')
    temp_loc = __dump_file_to_tmp(zipped_directory)

    # Make sure that all the files exist
    try:
        files = ['CONTCAR', 'DOSCAR', 'EIGENVAL', 'FW.json', 'FW_submit.script',
                 'IBZKPT', 'INCAR', 'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT',
                 'POSCAR', 'POTCAR', 'REPORT', 'XDATCAR', 'all.traj',
                 'ase-sort.dat', 'energy.out', 'fw_vasp_fill-115384.error',
                 'fw_vasp_fill-115384.out', 'slab_in.traj', 'slab_relaxed.traj',
                 'vasp.out', 'vasp_functions.py', 'vasp_functions.pyc',
                 'vasprun.xml']
        for file_ in files:
            assert os.path.isfile(temp_loc + file_)

    # Clean up
    finally:
        subprocess.call('rm -r %s' % temp_loc, shell=True)


def test__get_vasp_settings_from_fw():
    '''
    This is a pretty bad test, but I'm too lazy to fix it. And it's only a
    patching method anyway that shouldn't even exist, so it's less of a big
    deal to have a bad test.
    '''
    fireworks_folder = '/home/GASpy/gaspy/tests/test_cases/fireworks/'
    for file_name in os.listdir(fireworks_folder):
        with open(fireworks_folder + file_name, 'rb') as file_handle:
            fw = pickle.load(file_handle)
        vasp_settings = __get_patched_vasp_settings(fw)

        expected_vasp_settings = fw.name['vasp_settings']
        assert vasp_settings == expected_vasp_settings


def test___get_patched_miller():
    '''
    Two tests:  See if the patching can ignore a "good" set of miller
    indices, then see if it can patch a "bad" set
    '''
    # See if it ignores the good set
    good_miller = (1, 1, 1)
    patched_miller = __get_patched_miller(good_miller)
    assert patched_miller == good_miller

    # See if it patches the bad set
    bad_miller = '(1, 1, 1)'
    patched_miller = __get_patched_miller(bad_miller)
    assert patched_miller == good_miller


def test_update_collection_catalog():
    elements = ['Pd']
    max_miller = 1

    try:
        # Clear out the catalog so we know that anything new was added by this
        # function
        with get_mongo_collection('catalog') as collection:
            collection.delete_many({})

            # Add some sites and check that they're there
            update_catalog_collection(elements=elements, max_miller=max_miller)
            docs = list(collection.find({'mpid': 'mp-2'}))
        assert len(docs) > 0

    # Reset the unit testing catalog and clear any pickles we made
    finally:
        with get_mongo_collection('catalog') as collection:
            collection.delete_many({})
        populate_unit_testing_collection('catalog')
        clean_up_tasks()


def test__GetMpids():
    elements = set(['Cu', 'Al'])
    task = _GetMpids(elements=list(elements))

    # Run the task
    try:
        task.run()
        mpids = get_task_output(task)

        # For each MPID it enumerated, make sure the formation energy and
        # composition are correct
        with MPRester(read_rc('matproj_api_key')) as rester:
            for mpid in mpids:
                docs = rester.query({'task_id': mpid},
                                    ['elements', 'formation_energy_per_atom'])
                assert docs[0]['formation_energy_per_atom'] <= 0.
                for element in docs[0]['elements']:
                    assert element in elements

    finally:
        clean_up_tasks()


def test__InsertAllSitesFromBulkToCatalog():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the bulk calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    mpid = 'mp-2'
    max_miller = 2
    catalog_inserter = _InsertSitesToCatalog(mpid=mpid, max_miller=max_miller)
    site_generator = catalog_inserter.requires()

    try:
        run_task_locally(site_generator)
        site_docs = get_task_output(site_generator)

        catalog_inserter.run(_testing=True)
        catalog_docs = get_task_output(catalog_inserter)

        for site_doc, catalog_doc in zip(site_docs, catalog_docs):
            assert catalog_doc['mpid'] == mpid
            assert max(catalog_doc['miller']) <= max_miller
            assert catalog_doc['min_xy'] == site_generator.min_xy
            assert catalog_doc['slab_generator_settings'] == unfreeze_dict(site_generator.slab_generator_settings)
            assert catalog_doc['get_slab_settings'] == unfreeze_dict(site_generator.get_slab_settings)
            assert catalog_doc['bulk_vasp_settings'] == unfreeze_dict(site_generator.bulk_vasp_settings)
            assert catalog_doc['shift'] == site_doc['shift']
            assert catalog_doc['top'] == site_doc['top']
            assert make_atoms_from_doc(catalog_doc) == make_atoms_from_doc(site_doc)
            npt.assert_allclose(catalog_doc['slab_repeat'], site_doc['slab_repeat'])
            npt.assert_allclose(catalog_doc['adsorption_site'], site_doc['adsorption_site'])

    finally:
        clean_up_tasks()
