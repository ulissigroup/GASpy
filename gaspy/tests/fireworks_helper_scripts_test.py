'''
Tests for the `fireworks_helper_scripts` submodule.
'''

__authors__ = ['Aini Palizhati', 'Kevin Tran']
__emails__ = ['apalizha@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we are testing
from ..fireworks_helper_scripts import (get_launchpad,
                                        is_rocket_running,
                                        _get_firework_docs,
                                        __warn_about_fizzles,
                                        make_firework,
                                        encode_atoms_to_trajhex,
                                        decode_trajhex_to_atoms,
                                        submit_fwork,
                                        check_jobs_status,
                                        get_atoms_from_fwid,
                                        get_atoms_from_fw,
                                        __patch_old_atoms_tags)

# Things we need to do the tests
import pytest
import warnings
import pickle
import getpass
import pandas as pd
import ase
from fireworks import Firework, LaunchPad, FileWriteTask, PyTask, Workflow
from . import test_cases
from ..utils import read_rc
from .. import defaults

REGRESSION_BASELINES_LOCATION = ('/home/GASpy/gaspy/tests/regression_baselines'
                                 '/fireworks_helper_scripts/')
TEST_CASES_LOCATION = '/home/GASpy/gaspy/tests/test_cases/'
FIREWORKS_FOLDER = TEST_CASES_LOCATION + 'fireworks/'
FIREWORKS_FILES = [FIREWORKS_FOLDER + file_name for file_name in os.listdir(FIREWORKS_FOLDER)]


def test_get_launchpad():
    lpad = get_launchpad()
    assert isinstance(lpad, LaunchPad)
    assert hasattr(lpad, 'fireworks')

    # Make sure we fed the settings in correctly
    configs = read_rc('fireworks_info.lpad')
    configs['port'] = int(configs['port'])
    for key, value in configs.items():
        assert getattr(lpad, key) == value


def test_is_rocket_running():
    '''
    This test assumes that you have your FireWorks Mongo set up with a very
    specific set of documents inside a collection made strictly for unit
    testing. I am currently too lazy to document what those documents are
    in a systematic way. I figured the chances of someone like you actually
    caring were near-zero. If you do care, then email the code managers and
    we things up for you so you can run this test.
    '''
    # Let's test three things at once:
    # (1) can the function parse vasp settings correctly,
    # (2) can it report fizzles correctly, and
    # (3) will it tell us [correctly] if something is not running?
    query = {'fw_id': 353903}
    vasp_settings = {'kpts': [4, 4, 1],
                     'symprec': 1e-10,
                     'isym': 0,
                     'pp': 'PBE',
                     'encut': 350,
                     'pp_version': '5.4',
                     'isif': 0,
                     'ibrion': 2,
                     'gga': 'RP',
                     'ediffg': -0.03,
                     'nsw': 200,
                     'lreal': 'Auto'}
    with warnings.catch_warnings(record=True) as warning_manager:
        warnings.simplefilter('always')
        assert is_rocket_running(query, vasp_settings) is False
        assert len(warning_manager) == 1
        assert issubclass(warning_manager[-1].category, RuntimeWarning)
        assert 'We have fizzled a calculation' in str(warning_manager[-1].message)

    # Test if it can correctly flag a bunch of running rockets
    for fwid in [365912, 355429, 369302, 355479, 365912]:
        assert is_rocket_running({'fw_id': fwid}, {}) is True


def test__get_firework_docs():
    query = {'fw_id': 353903}
    docs = _get_firework_docs(query, _testing=True)
    assert len(docs) == 1
    assert isinstance(docs[0], dict)
    assert docs[0]['fw_id'] == query['fw_id']


def test___warn_about_fizzles():
    with warnings.catch_warnings(record=True) as warning_manager:
        warnings.simplefilter('always')

        # When things are not fizzled, make sure it says nothing
        docs = [{'state': 'COMPLETED', 'fw_id': 0}]
        __warn_about_fizzles(docs)
        assert len(warning_manager) == 0

        # When things are fizzled, make sure it warns us
        docs = [{'state': 'FIZZLED', 'fw_id': 1}]
        __warn_about_fizzles(docs)
        assert len(warning_manager) == 1
        assert issubclass(warning_manager[-1].category, RuntimeWarning)
        assert 'We have fizzled a calculation' in str(warning_manager[-1].message)


def test_make_firework():
    '''
    Our FireWork rockets should take three steps:  write our
    `vasp_functions.py` submodule to the local directory, write the atoms
    object to the local directory, and then perform the VASP relaxation. We'll
    pick apart each of these during this test.
    '''
    # Make the firework and pull out the operations so we can inspect them
    atoms = ase.Atoms('CO')
    fw_name = {'calculation_type': 'gas phase optimization', 'gasname': 'CO'}
    vasp_settings = defaults.gas_settings()['vasp']
    fwork = make_firework(atoms, fw_name, vasp_settings)
    pass_vasp_functions, read_atoms_file, relax = fwork.tasks

    # Make sure it's actually a Firework object and its name is correct
    assert isinstance(fwork, Firework)
    fw_name_expected = fw_name.copy()
    fw_name_expected['user'] == getpass.getuser()
    assert fwork.name == fw_name_expected

    # Make sure vasp_function.py passes correctly
    assert isinstance(pass_vasp_functions, FileWriteTask)
    assert pass_vasp_functions['files_to_write'][0]['filename'] == 'vasp_functions.py'
    with open('/home/GASpy/gaspy/vasp_functions.py') as file_handle:
        expected_vasp_functions_contents = file_handle.read()
    assert (pass_vasp_functions['files_to_write'][0]['contents'] ==
            expected_vasp_functions_contents)

    # Make sure we read our atoms object correctly
    assert isinstance(read_atoms_file, PyTask)
    assert read_atoms_file['func'] == 'vasp_functions.hex_to_file'
    assert read_atoms_file['args'] == ['slab_in.traj', encode_atoms_to_trajhex(atoms)]

    # Make sure we start VASP
    assert isinstance(relax, PyTask)
    assert relax['func'] == 'vasp_functions.runVasp'
    assert relax['args'] == ['slab_in.traj', 'slab_relaxed.traj', vasp_settings]
    assert relax['stored_data_varname'] == 'opt_results'


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'O_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_encode_atoms_to_trajhex(adslab_atoms_name):
    '''
    This actually tests GASpy's ability to both encode and decode, because what
    we really care about is being able to successfully decode whatever we
    encode.

    This is hard-coded for adslabs. It should be able to work on bulks and
    slabs, too.  Feel free to update it.
    '''
    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)

    trajhex = encode_atoms_to_trajhex(expected_atoms)
    atoms = decode_trajhex_to_atoms(trajhex)
    assert atoms == expected_atoms


@pytest.mark.baseline
@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'O_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_to_create_atoms_trajhex_encoding(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and
    slabs, too.  Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    hex_ = encode_atoms_to_trajhex(atoms)

    file_name = (REGRESSION_BASELINES_LOCATION + 'trajhex_for_' +
                 adslab_atoms_name.split('.')[0] + '.pkl')
    with open(file_name, 'wb') as file_handle:
        pickle.dump(hex_, file_handle)
    assert True


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'O_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_decode_trajhex_to_atoms(adslab_atoms_name):
    '''
    This is a regression test to make sure that we can keep reading old hex
    strings and turning them into the appropriate atoms objects. We should
    probably start testing whether we can find the different indexes, but
    that's for future us to worry about.

    This is hard-coded for adslabs. It should be able to work on bulks and
    slabs, too.  Feel free to update it.
    '''
    file_name = (REGRESSION_BASELINES_LOCATION + 'trajhex_for_' +
                 adslab_atoms_name.split('.')[0] + '.pkl')
    with open(file_name, 'rb') as file_handle:
        trajhex = pickle.load(file_handle)
    atoms = decode_trajhex_to_atoms(trajhex)

    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    assert atoms == expected_atoms


def test_submit_fwork():
    atoms = ase.Atoms('CO')
    fw_name = {'calculation_type': 'gas phase optimization', 'gasname': 'CO'}
    vasp_settings = defaults.gas_settings()['vasp']
    fwork = make_firework(atoms, fw_name, vasp_settings)
    wflow = submit_fwork(fwork, _testing=True)
    assert len(wflow.fws) == 1
    assert isinstance(wflow, Workflow)
    assert wflow.name == 'vasp optimization'


@pytest.mark.parametrize('fw_file', FIREWORKS_FILES)
def test_get_atoms_from_fwid(fw_file):
    fwid = int(fw_file.split('.')[0].split('/')[-1])
    atoms = get_atoms_from_fwid(fwid)
    assert isinstance(atoms, ase.Atoms)


@pytest.mark.parametrize('fw_file', FIREWORKS_FILES)
def test__get_atoms_from_fw(fw_file):
    with open(fw_file, 'rb') as file_handle:
        fw = pickle.load(file_handle)
    atoms = get_atoms_from_fw(fw)
    assert isinstance(atoms, ase.Atoms)


@pytest.mark.parametrize('fw_file', FIREWORKS_FILES)
def test___patch_old_atoms(fw_file):
    with open(fw_file, 'rb') as file_handle:
        fw = pickle.load(file_handle)
    atoms = get_atoms_from_fw(fw)
    patched_atoms = __patch_old_atoms_tags(fw, atoms)

    # Make sure things are tagged correctly
    adsorbate_name = fw.name['adsorbate']
    for atom in patched_atoms:
        tag = atom.tag
        if tag == 0:
            assert atom.symbol not in adsorbate_name
        elif tag == 1:
            assert atom.symbol in adsorbate_name


@pytest.mark.parametrize('user, n_jobs',
                         [('zulissi', 10),
                          ('apalizha', 10),
                          ('zulissi', 20)])
def test_check_jobs_user(user, n_jobs):
    ''' This function test if the DataFrame contains only the user inquired '''
    dataframe = check_jobs_status(user, n_jobs)

    assert isinstance(dataframe, pd.DataFrame)
    assert user == dataframe['user'].unique()
    assert n_jobs == len(dataframe.index)
