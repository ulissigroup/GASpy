''' Tests for the `utils` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ..utils import (read_rc,
                     _find_rc_file,
                     fingerprint_atoms,
                     find_adsorption_sites,
                     unfreeze_dict,
                     encode_atoms_to_hex,
                     decode_hex_to_atoms,
                     encode_atoms_to_trajhex,
                     decode_trajhex_to_atoms,
                     save_luigi_task_run_results,
                     evaluate_luigi_task)

# Things we need to do the tests
import pytest
import pickle
import collections
import json
import numpy as np
import numpy.testing as npt
import luigi
from luigi.parameter import _FrozenOrderedDict
from . import test_cases
from .. import defaults
from ..mongo import make_atoms_from_doc
from ..gasdb import get_mongo_collection

REGRESSION_BASELINES_LOCATION = '/home/GASpy/gaspy/tests/regression_baselines/utils/'
TASKS_OUTPUTS_LOCATION = read_rc('gasdb_path')


@pytest.mark.parametrize('query',
                         [None,
                          'gasdb_path',
                          'lpad.host',
                          'mongo_info.atoms.port'])
def test_read_rc(query):
    rc_contents = read_rc(query)

    # Nearly identical to the `read_rc` function's method
    # of getting the content, but without the fluff and hardcoded
    with open('/home/GASpy/gaspy/tests/.gaspyrc.json', 'r') as file_handle:
        expected_rc_contents = json.load(file_handle)
    if query:
        keys = query.split('.')
        for key in keys:
            expected_rc_contents = expected_rc_contents[key]

    assert rc_contents == expected_rc_contents


def test_read_rc_exception():
    '''
    The `read_rc` function should be raising a specific exception when users
    provide the wrong keys. This is the test to make sure that happens.
    '''
    try:
        _ = read_rc(query='this.should.not.work')  # noqa: F841

        # If there are no errors at all, something went wrong
        assert False

    # If there was a KeyError but it wasn't the one it was supposed to be,
    # then something went wrong
    except KeyError as error:
        if str(error) != "'Check the spelling/capitalization of the key/values you are looking for'":
            assert False

        # Pass if the message is correct
        else:
            assert True


def test__find_rc_file():
    '''
    This test assumes that you have already inserted the
    /home/GASpy/tests/ folder to the front of your PYTHONPATH,
    which should mean that the rc file that it finds should be the
    one in the testing folder.
    '''
    rc_file = _find_rc_file()
    expected_rc_file = '/home/GASpy/gaspy/tests/.gaspyrc.json'
    assert rc_file == expected_rc_file


@pytest.mark.baseline
@pytest.mark.parametrize('collection_tag', ['catalog', 'adsorption'])
def test_to_create_fingerprints(collection_tag):
    # Get and fingerprint the documents
    with get_mongo_collection(collection_tag) as collection:
        docs = list(collection.find())
    fingerprints = []
    for doc in docs:
        atoms = make_atoms_from_doc(doc)
        fingerprint = fingerprint_atoms(atoms)
        fingerprints.append(fingerprint)

    # Save them
    cache_location = REGRESSION_BASELINES_LOCATION + 'fingerprints_of_%s' % collection_tag + '.json'
    with open(cache_location, 'w') as file_handle:
        json.dump(fingerprints, file_handle)


@pytest.mark.parametrize('collection_tag', ['catalog', 'adsorption'])
def test_fingerprint_atoms(collection_tag):
    # Load the cache of baseline answers
    cache_location = REGRESSION_BASELINES_LOCATION + 'fingerprints_of_%s' % collection_tag + '.json'
    with open(cache_location, 'r') as file_handle:
        expected_fingerprints = json.load(file_handle)

    # Get and fingerprint the documents, then test
    with get_mongo_collection(collection_tag) as collection:
        docs = list(collection.find())
    for doc, expected_fingerprint in zip(docs, expected_fingerprints):
        atoms = make_atoms_from_doc(doc)
        fingerprint = fingerprint_atoms(atoms)
        assert fingerprint == expected_fingerprint


@pytest.mark.baseline
@pytest.mark.parametrize('slab_atoms_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_to_create_adsorption_sites(slab_atoms_name):
    atoms = test_cases.get_slab_atoms(slab_atoms_name)
    sites = find_adsorption_sites(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'sites_for_' + slab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(sites, file_handle)
    assert True


@pytest.mark.parametrize('slab_atoms_name',
                         ['AlAu2Cu_210.traj',
                          'CoSb2_110.traj',
                          'Cu_211.traj',
                          'FeNi_001.traj',
                          'Ni4W_001.traj',
                          'Pt12Si5_110.traj'])
def test_find_adsorption_sites(slab_atoms_name):
    '''
    Check out `.learning_tests.pymatgen_test._get_sites_for_standard_structure`
    to see what pymatgen gives us. Our `gaspy.utils.find_adsorption_sites` simply gives us
    the value of that object when the key is 'all'.
    '''
    atoms = test_cases.get_slab_atoms(slab_atoms_name)
    sites = find_adsorption_sites(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'sites_for_' + slab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        expected_sites = pickle.load(file_handle)

    npt.assert_allclose(np.array(sites), np.array(expected_sites), rtol=1e-5, atol=1e-7)


def test_unfreeze_dict():
    frozen_dict = _FrozenOrderedDict(foo='bar', bar=('foo', 'bar'),
                                     sub_dict0=_FrozenOrderedDict(),
                                     sub_dict1=_FrozenOrderedDict(foo=['']),
                                     sub_dict2=dict(foo=True,
                                                    bar=_FrozenOrderedDict(foo=1.0),
                                                    array=['foo', _FrozenOrderedDict(foo='bar')]))
    unfrozen_dict = unfreeze_dict(frozen_dict)
    _look_for_type_in_dict(type_=_FrozenOrderedDict, dict_=unfrozen_dict)


def _look_for_type_in_dict(type_, dict_):
    '''
    Recursive function that checks if there is any object type inside any branch
    of a dictionary. It does so by performing an `assert` check on every single
    value in the dictionary. Note that we could use EAFP instead of if/then checking,
    but that ended up being very illegible.

    Args:
        type_   An object type (e.g, int, float, str, etc) that you want to look for
        dict_   A dictionary that you want to parse. Can really be any object with
                the `items` method.
    '''
    # Check this part of the dictionary branch
    assert not isinstance(dict_, type_)

    # If this branch is a dictionary, then recur on the values of the dictionary
    if isinstance(dict_, collections.Mapping):
        for key, value in dict_.items():
            _look_for_type_in_dict(type_, value)

    # If this branch is iterable, then recur on each element in the iterable
    elif isinstance(dict_, collections.Iterable) and not isinstance(dict_, str):
        for element in dict_:
            _look_for_type_in_dict(type_, element)


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_encode_atoms_to_hex(adslab_atoms_name):
    '''
    This actually tests GASpy's ability to both encode and decode,
    because what we really care about is being able to successfully decode whatever
    we encode.

    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)

    hex_ = encode_atoms_to_hex(expected_atoms)
    atoms = decode_hex_to_atoms(hex_)
    assert atoms == expected_atoms


def test_decode_hex_to_atoms():
    '''
    This is a regression test to make sure that we can keep reading old hex strings
    and turning them into the appropriate atoms objects.

    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    expected_atoms = defaults.adsorbates_dict()['CO']

    # Example hex from GASpy v0.1
    hex_ = '63636f70795f7265670a5f7265636f6e7374727563746f720a70310a28636173652e61746f6d730a41746f6d730a70320a635f5f6275696c74696e5f5f0a6f626a6563740a70330a4e745270340a286470350a5327696e666f270a70360a286470370a7353275f63656c6c64697370270a70380a636e756d70792e636f72652e6d756c746961727261790a5f7265636f6e7374727563740a70390a28636e756d70790a6e6461727261790a7031300a2849300a74532762270a74527031310a2849310a2849330a49310a74636e756d70790a64747970650a7031320a2853276638270a49300a49310a74527031330a2849330a53273c270a4e4e4e492d310a492d310a49300a74624930300a53275c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c783030270a74627353275f63616c63270a7031340a4e735327617272617973270a7031350a28647031360a5327706f736974696f6e73270a7031370a67390a286731300a2849300a74532762270a74527031380a2849310a2849320a49330a746731330a4930300a53275c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830303333333333335c7866333f270a74627353276e756d62657273270a7031390a67390a286731300a2849300a74532762270a74527032300a2849310a2849320a746731320a2853276938270a49300a49310a74527032310a2849330a53273c270a4e4e4e492d310a492d310a49300a74624930300a53275c7830365c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830385c7830305c7830305c7830305c7830305c7830305c7830305c783030270a7462737353275f706263270a7032320a67390a286731300a2849300a74532762270a74527032330a2849310a2849330a746731320a2853276231270a49300a49310a74527032340a2849330a53277c270a4e4e4e492d310a492d310a49300a74624930300a53275c7830305c7830305c783030270a74627353275f63656c6c270a7032350a67390a286731300a2849300a74532762270a74527032360a2849310a2849330a49330a746731330a4930300a53275c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c7830305c783030270a74627353275f636f6e73747261696e7473270a7032370a286c7032380a73622e'
    atoms = decode_hex_to_atoms(hex_)
    assert atoms == expected_atoms

    # Example hex from GASpy v0.2
    hex_ = '8003636173652e61746f6d730a41746f6d730a7100298171017d7102285804000000696e666f71037d710458090000005f63656c6c646973707105636e756d70792e636f72652e6d756c746961727261790a5f7265636f6e7374727563740a7106636e756d70790a6e6461727261790a71074b00857108430162710987710a52710b284b014b034b0186710c636e756d70790a64747970650a710d58020000006638710e4b004b0187710f527110284b0358010000003c71114e4e4e4affffffff4affffffff4b007471126289431800000000000000000000000000000000000000000000000071137471146258050000005f63616c6371154e580600000061727261797371167d7117285809000000706f736974696f6e737118680668074b00857119680987711a52711b284b014b024b0386711c681089433000000000000000000000000000000000000000000000000000000000000000000000000000000000333333333333f33f711d74711e6258070000006e756d62657273711f680668074b008571206809877121527122284b014b02857123680d5802000000693871244b004b01877125527126284b0368114e4e4e4affffffff4affffffff4b0074712762894310060000000000000008000000000000007128747129627558040000005f706263712a680668074b0085712b680987712c52712d284b014b0385712e680d58020000006231712f4b004b01877130527131284b0358010000007c71324e4e4e4affffffff4affffffff4b007471336289430300000071347471356258050000005f63656c6c7136680668074b008571376809877138527139284b014b034b0386713a6810894348000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000713b74713c62580c0000005f636f6e73747261696e7473713d5d713e75622e'
    atoms = decode_hex_to_atoms(hex_)
    assert atoms == expected_atoms


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_encode_atoms_to_trajhex(adslab_atoms_name):
    '''
    This actually tests GASpy's ability to both encode and decode,
    because what we really care about is being able to successfully decode whatever
    we encode.

    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)

    trajhex = encode_atoms_to_trajhex(expected_atoms)
    atoms = decode_trajhex_to_atoms(trajhex)
    assert atoms == expected_atoms


@pytest.mark.baseline
@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_to_create_atoms_trajhex_encoding(adslab_atoms_name):
    '''
    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    hex_ = encode_atoms_to_trajhex(atoms)

    file_name = REGRESSION_BASELINES_LOCATION + 'trajhex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'wb') as file_handle:
        pickle.dump(hex_, file_handle)
    assert True


@pytest.mark.parametrize('adslab_atoms_name',
                         ['CO_dissociate_Pt12Si5_110.traj',
                          'CO_top_Cu_211.traj',
                          'C_hollow_AlAu2Cu_210.traj',
                          'OH_desorb_CoSb2_110.traj',
                          'OOH_dissociate_Ni4W_001.traj',
                          'OOH_hollow_FeNi_001.traj'])
def test_decode_trajhex_to_atoms(adslab_atoms_name):
    '''
    This is a regression test to make sure that we can keep reading old hex strings
    and turning them into the appropriate atoms objects.

    This is hard-coded for adslabs. It should be able to work on bulks and slabs, too.
    Feel free to update it.
    '''
    file_name = REGRESSION_BASELINES_LOCATION + 'trajhex_for_' + adslab_atoms_name.split('.')[0] + '.pkl'
    with open(file_name, 'rb') as file_handle:
        trajhex = pickle.load(file_handle)
    atoms = decode_trajhex_to_atoms(trajhex)

    expected_atoms = test_cases.get_adslab_atoms(adslab_atoms_name)
    assert atoms == expected_atoms


def test_save_luigi_task_run_results():
    '''
    Instead of actually testing this function, we perform a rough
    learning test on Luigi.
    '''
    assert 'temporary_path' in dir(luigi.LocalTarget)


def test_evaluate_luigi_task():
    '''
    We made some test tasks and try to execute them here. Then we verify
    the output results of the tasks.
    '''
    # Define where/what the outputs should be
    output_file_names = ['BranchTestTask/BranchTestTask_False_1_ca4048d8e6.pkl',
                         'BranchTestTask/BranchTestTask_False_42_fedcdcbd62.pkl',
                         'BranchTestTask/BranchTestTask_True_7_498ea8eed2.pkl',
                         'RootTestTask/RootTestTask__99914b932b.pkl']
    output_file_names = [TASKS_OUTPUTS_LOCATION + '/pickles/' + file_name
                         for file_name in output_file_names]
    expected_outputs = [1, 42, 7, 'We did it!']

    # Run the tasks
    try:
        evaluate_luigi_task(RootTestTask())

        # Test that each task executed correctly
        for output_file_name, expected_output in zip(output_file_names, expected_outputs):
            with open(output_file_name, 'rb') as file_handle:
                output = pickle.load(file_handle)
            assert output == expected_output

        # Test that when the "force" argument is `False`, tasks ARE NOT rerun
        file_creation_times = [os.path.getmtime(output_file) for output_file in output_file_names]
        evaluate_luigi_task(RootTestTask(), force=False)
        for output_file, expected_ctime in zip(output_file_names, file_creation_times):
            ctime = os.path.getmtime(output_file)
            assert ctime == expected_ctime
        # Test that when the "force" argument is `True`, tasks ARE rerun
        evaluate_luigi_task(RootTestTask(), force=True)
        for output_file, old_ctime in zip(output_file_names, file_creation_times):
            ctime = os.path.getmtime(output_file)
            assert ctime > old_ctime

        # Clean up, regardless of what happened during testing
        __delete_files(output_file_names)
    except: # noqa: 722
        __delete_files(output_file_names)
        raise


def __delete_files(file_names):
    ''' Helper function to try and delete some files '''
    for file_name in file_names:
        try:
            os.remove(file_name)
        except OSError:
            pass


class RootTestTask(luigi.Task):
    def requires(self):
        return [BranchTestTask(task_result=1),
                BranchTestTask(task_result=7, branch_again=True)]

    def run(self):
        save_luigi_task_run_results(self, 'We did it!')

    def output(self):
        return luigi.LocalTarget(TASKS_OUTPUTS_LOCATION + '/pickles/%s/%s.pkl'
                                 % (type(self).__name__, self.task_id))


class BranchTestTask(luigi.Task):
    task_result = luigi.IntParameter(42)
    branch_again = luigi.BoolParameter(False)

    def requires(self):
        if self.branch_again:
            return BranchTestTask()
        else:
            return

    def run(self):
        save_luigi_task_run_results(self, self.task_result)

    def output(self):
        return luigi.LocalTarget(TASKS_OUTPUTS_LOCATION + '/pickles/%s/%s.pkl'
                                 % (type(self).__name__, self.task_id))
