''' Tests for the `gaspy.tasks.calculation_finders` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.calculation_finders import (FindGas,
                                          _find_docs_in_atoms_collection,
                                          _remove_old_docs)

# Things we need to do the tests
import pytest
import warnings
from bson.objectid import ObjectId
import luigi
from .utils import clean_up_task
from ... import defaults
from ...tasks.core import get_task_output


def test_FindGas_successfully():
    '''
    If we ask this task to find something that is there, it should return it
    '''
    vasp_settings = defaults.GAS_SETTINGS['vasp']
    task = FindGas('CO', vasp_settings)

    try:
        warnings.simplefilter('ignore')
        task.run()
        doc = get_task_output(task)
        assert doc['type'] == 'gas'
        assert doc['fwname']['gasname'] == 'CO'

        for key, value in vasp_settings.items():
            try:
                assert doc['fwname']['vasp_settings'][key] == value
            # Some of our VASP settings are tuples, but Mongo only saves lists.
            # If we're looking at one of these cases, then we should compare
            # list-to-list
            except AssertionError:
                if isinstance(value, tuple):
                    assert doc['fwname']['vasp_settings'][key] == list(value)
                else:
                    raise

    finally:
        clean_up_task(task)


def test_FindGas_unsuccessfully():
    task = FindGas('N2', defaults.GAS_SETTINGS['vasp'])

    try:
        dependency = task.run()
        assert isinstance(dependency, luigi.Task)

    finally:
        clean_up_task(task)


def test__find_docs_in_atoms_collection():
    '''
    We should probably have more test cases than this.
    But I'm too lazy right now.
    '''
    query = {'type': 'gas', 'fwname.gasname': 'CO'}
    vasp_settings = {'kpts': [1, 1, 1],
                     'xc': 'beef-vdw',
                     'encut': 350,
                     'isif': 0,
                     'ibrion': 2,
                     'ediffg': -0.03,
                     'nsw': 100,
                     'pp_version': '5.3.5',
                     'pp_guessed': True,
                     'pp': 'PBE',
                     'gga': 'BF',
                     'luse_vdw': True,
                     'zab_vdw': -1.8867,
                     'lbeefens': True}
    docs = _find_docs_in_atoms_collection(query, vasp_settings)
    assert len(docs) == 1
    assert docs[0]['_id'] == ObjectId('5b84e3190b0c9ee53cfc2848')


def test__remove_old_docs():
    '''
    This could be three tests, but I bunched them into one.
    '''
    # If there's only one document, we should return it
    docs = ['foo']
    doc = _remove_old_docs(docs)
    assert doc == 'foo'

    # If there's two, make sure we get the new ond and get an error
    docs = [{'fwid': 3, 'foo': 'bar'},
            {'fwid': 1, 'should stay': False}]
    with warnings.catch_warnings(record=True) as warning_manager:
        warnings.simplefilter('always')
        doc = _remove_old_docs(docs)
        assert len(warning_manager) == 1
        assert issubclass(warning_manager[-1].category, RuntimeWarning)
        assert 'We will be using the latest one, 3' in str(warning_manager[-1].message)

    # If there's nothing, make sure we get an error
    docs = []
    with pytest.raises(SyntaxError, message='Expected a RuntimeError') as exc_info:
        doc = _remove_old_docs(docs)
        assert ('You tried to parse out old documents, but did not pass any'
                in str(exc_info.value))
