''' Tests for the `gaspy.tasks.calculation_finders` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.calculation_finders import (FindCalculation,
                                          CalculationNotFoundError,
                                          FindGas,
                                          FindBulk,
                                          FindAdslab)

# Things we need to do the tests
import pytest
import warnings
import math
import luigi
from .utils import clean_up_task
from ... import defaults
from ...utils import turn_site_into_str, unfreeze_dict
from ...mongo import make_atoms_from_doc
from ...tasks.core import get_task_output
from ...tasks.make_fireworks import (MakeGasFW,
                                     MakeBulkFW,
                                     MakeAdslabFW)


def test_FindCalculation():
    '''
    We do a very light test of this parent class, because we will rely more
    heavily on the testing on the child classes and methods.
    '''
    finder = FindCalculation()
    assert isinstance(finder, luigi.Task)
    assert hasattr(finder, 'run')
    assert hasattr(finder, 'output')


def test_CalculationNotFoundError():
    assert isinstance(CalculationNotFoundError(), ValueError)


def test__remove_old_docs():
    '''
    This could be three tests, but I bunched them into one.
    '''
    remove_old_docs = FindCalculation()._remove_old_docs

    # If there's only one document, we should return it
    docs = ['foo']
    doc = remove_old_docs(docs)
    assert doc == 'foo'

    # If there's two, make sure we get the new ond and get an error
    docs = [{'fwid': 3, 'foo': 'bar'},
            {'fwid': 1, 'should stay': False}]
    with warnings.catch_warnings(record=True) as warning_manager:
        warnings.simplefilter('always')
        doc = remove_old_docs(docs)
        assert len(warning_manager) == 1
        assert issubclass(warning_manager[-1].category, RuntimeWarning)
        assert 'We will be using the latest one, 3' in str(warning_manager[-1].message)

    # If there's nothing, make sure we get an error
    docs = []
    with pytest.raises(CalculationNotFoundError) as exc_info:
        doc = remove_old_docs(docs)
        assert ('You tried to parse out old documents, but did not pass any'
                in str(exc_info.value))


def _assert_vasp_settings(doc, vasp_settings):
    '''
    Asserts whether the vasp_settings inside a doc/dictionary are correct

    Args:
        doc             Dictionary/Mongo document object
        vasp_settings   Dictionary of VASP settings
    '''
    for key, value in vasp_settings.items():
        try:
            assert doc['fwname']['vasp_settings'][key] == value

        # Some of our VASP settings are tuples, but Mongo only saves lists.
        # If we're looking at one of these cases, then we should compare
        # list-to-list
        except AssertionError:
            if isinstance(value, tuple):
                assert doc['fwname']['vasp_settings'][key] == list(value)

        except KeyError:
            # If we're looking at an adslab, then we don't care about certain
            # vasp settings
            if doc['type'] == 'slab+adsorbate' and key in set(['nsw', 'isym', 'symprec']):
                pass

            # If we're looking at a slab, then we don't care about certain
            # vasp settings
            elif doc['type'] == 'slab+adsorbate' and key in set(['isym']):
                pass

            else:
                raise


def test_FindGas_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    gas = 'H2'
    vasp_settings = defaults.GAS_SETTINGS['vasp']
    task = FindGas(gas, vasp_settings)

    try:
        _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)
        assert doc['type'] == 'gas'
        assert doc['fwname']['gasname'] == gas
        _assert_vasp_settings(doc, vasp_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_task(task)


def _run_task_with_dynamic_dependencies(task):
    '''
    If a task has dynamic dependencies, then it will return a generator. This
    function will run the task for you, iterate through the generator, and
    return the results.
    '''
    try:
        output = next(task.run(_testing=True))
        return output
    except StopIteration:
        pass


def test_FindGas_unsuccessfully():
    '''
    If we ask this task to find something that is not there, it should return
    the correct dependency
    '''
    gas = 'CHO'
    task = FindGas(gas, defaults.GAS_SETTINGS['vasp'])

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeGasFW)
        assert dependency.gas_name == gas

    finally:
        clean_up_task(task)


def test_FindBulk_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    mpid = 'mp-2'
    vasp_settings = defaults.BULK_SETTINGS['vasp']
    task = FindBulk(mpid, vasp_settings)

    try:
        # Some weird testing interactions mean that this task might already be
        # done. If that happens, then just delete the output and try again
        try:
            _run_task_with_dynamic_dependencies(task)
        except luigi.target.FileAlreadyExists:
            clean_up_task(task)
            _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)

        assert doc['type'] == 'bulk'
        assert doc['fwname']['mpid'] == mpid
        _assert_vasp_settings(doc, vasp_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_task(task)


def test_FindBulk_unsuccessfully():
    '''
    If we ask this task to find something that is not there, it should return
    the correct dependency
    '''
    mpid = 'mp-120'
    task = FindBulk(mpid, defaults.BULK_SETTINGS['vasp'])

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeBulkFW)
        assert dependency.mpid == mpid

    finally:
        clean_up_task(task)


def test_FindAdslab_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    adsorption_site = (0., 1.41, 20.52)
    shift = 0.25
    top = True
    adsorbate_name = 'CO'
    rotation = {'phi': 0., 'theta': 0., 'psi': 0.}
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    vasp_settings = defaults.ADSLAB_SETTINGS['vasp']
    task = FindAdslab(adsorption_site=adsorption_site,
                      shift=shift,
                      top=top,
                      adsorbate_name=adsorbate_name,
                      rotation=rotation,
                      mpid=mpid,
                      miller_indices=miller_indices,
                      vasp_settings=vasp_settings)

    try:
        _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)
        assert doc['type'] == 'slab+adsorbate'
        assert doc['fwname']['adsorption_site'] == turn_site_into_str(adsorption_site)
        assert math.isclose(doc['fwname']['shift'], shift)
        assert doc['fwname']['top'] == top
        assert doc['fwname']['adsorbate'] == adsorbate_name
        assert doc['fwname']['adsorbate_rotation'] == rotation
        assert doc['fwname']['mpid'] == mpid
        assert tuple(doc['fwname']['miller']) == miller_indices
        _assert_vasp_settings(doc, vasp_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_task(task)


def test_FindAdslab_unsuccessfully():
    '''
    If we ask this task to find something that is not there, it should return
    the correct dependency
    '''
    adsorption_site = (0., 1.41, 20.52)
    shift = 0.25
    top = True
    adsorbate_name = 'OOH'
    rotation = {'phi': 0., 'theta': 0., 'psi': 0.}
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    vasp_settings = defaults.ADSLAB_SETTINGS['vasp']
    task = FindAdslab(adsorption_site=adsorption_site,
                      shift=shift,
                      top=top,
                      adsorbate_name=adsorbate_name,
                      rotation=rotation,
                      mpid=mpid,
                      miller_indices=miller_indices,
                      vasp_settings=vasp_settings)

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeAdslabFW)
        assert dependency.mpid == mpid
        assert dependency.adsorption_site == adsorption_site
        assert dependency.shift == shift
        assert dependency.top == top
        assert dependency.adsorbate_name == adsorbate_name
        assert unfreeze_dict(dependency.rotation) == rotation
        assert dependency.mpid == mpid
        assert unfreeze_dict(dependency.vasp_settings) == vasp_settings

    finally:
        clean_up_task(task)
