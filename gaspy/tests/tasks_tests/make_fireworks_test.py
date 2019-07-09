''' Tests for the `gaspy.tasks.make_fireworks` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

import warnings
warnings.filterwarnings('ignore', category=ImportWarning)

# Things we're testing
from ...tasks.make_fireworks import (FireworkMaker,
                                     MakeGasFW,
                                     MakeBulkFW,
                                     MakeAdslabFW,
                                     MakeSurfaceFW)

# Things we need to do the tests
import pytest
import luigi
from .utils import clean_up_tasks, run_task_locally
from ... import defaults
from ...utils import unfreeze_dict
from ...tasks.core import get_task_output, schedule_tasks
from ...tasks.calculation_finders import FindSurface
from ...tasks.atoms_generators import (GenerateGas,
                                       GenerateBulk,
                                       GenerateAdslabs)

GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


def test_FireworkMaker():
    assert issubclass(FireworkMaker, luigi.Task)
    assert FireworkMaker().complete() is False


def test_MakeGasFW():
    clean_up_tasks()
    gas_name = 'CO'
    task = MakeGasFW(gas_name, GAS_SETTINGS['vasp'])

    # Check that our requirment is correct
    req = task.requires()
    assert isinstance(req, GenerateGas)
    assert req.gas_name == task.gas_name

    try:
        # Need to make sure our requirement is run before testing our task
        req.run()

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['calculation_type'] == 'gas phase optimization'
        assert fwork.name['gasname'] == gas_name
        assert fwork.name['dft_settings'] == GAS_SETTINGS['vasp']
        assert task.complete() is True

    finally:
        clean_up_tasks()


def test_MakeBulkFW():
    clean_up_tasks()
    mpid = 'mp-30'
    task = MakeBulkFW(mpid, BULK_SETTINGS['vasp'])

    # Check that our requirment is correct
    req = task.requires()
    assert isinstance(req, GenerateBulk)
    assert req.mpid == task.mpid

    try:
        # Need to make sure our requirement is run before testing our task
        req.run()

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['calculation_type'] == 'unit cell optimization'
        assert fwork.name['mpid'] == mpid
        assert fwork.name['dft_settings'] == BULK_SETTINGS['vasp']
        assert task.complete() is True

    finally:
        clean_up_tasks()


def test_MakeBulkFW_max_size():
    clean_up_tasks()
    mpid = 'mp-30'
    task = MakeBulkFW(mpid, BULK_SETTINGS['vasp'], max_atoms=0)

    try:
        # Need to make sure our requirement is run before testing our task
        req = task.requires()
        req.run()

        # Manually call the `run` method with the unit testing flag to trigger
        # the error
        with pytest.raises(ValueError, match='The size of the bulk'):
            _ = task.run(_testing=True)     # noqa: F841

    finally:
        clean_up_tasks()


def test_MakeAdslabFW():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make an Adslab from a bulk that shows up in the unit_testing_atoms Mongo
    collection. If you copy/paste this test into somewhere else, make sure
    that you use `run_task_locally` appropriately.
    '''
    clean_up_tasks()
    adsorption_site = (1.48564485e-23, 1.40646118e+00, 2.08958465e+01)
    shift = 0.25
    top = False
    dft_settings = ADSLAB_SETTINGS['vasp']
    adsorbate_name = 'OH'
    rotation = ADSLAB_SETTINGS['rotation']
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    min_xy = ADSLAB_SETTINGS['min_xy']
    slab_generator_settings = SLAB_SETTINGS['slab_generator_settings']
    get_slab_settings = SLAB_SETTINGS['get_slab_settings']
    bulk_dft_settings = BULK_SETTINGS['vasp']
    task = MakeAdslabFW(adsorption_site=adsorption_site,
                        shift=shift,
                        top=top,
                        dft_settings=dft_settings,
                        adsorbate_name=adsorbate_name,
                        rotation=rotation,
                        mpid=mpid,
                        miller_indices=miller_indices,
                        min_xy=min_xy,
                        slab_generator_settings=slab_generator_settings,
                        get_slab_settings=get_slab_settings,
                        bulk_dft_settings=bulk_dft_settings)

    # Check that our requirement is correct
    req = task.requires()
    assert isinstance(req, GenerateAdslabs)
    assert req.adsorbate_name == adsorbate_name
    assert unfreeze_dict(req.rotation) == rotation
    assert req.mpid == mpid
    assert req.miller_indices == miller_indices
    assert req.min_xy == min_xy
    assert unfreeze_dict(req.slab_generator_settings) == slab_generator_settings
    assert unfreeze_dict(req.get_slab_settings) == get_slab_settings
    assert unfreeze_dict(req.bulk_dft_settings) == bulk_dft_settings

    try:
        # Need to make sure our requirement is run before testing our task.
        # Make sure you are asking for something that does not need a FireWork
        # submission, i.e., make sure the requirements are already in the
        # unit testing Mongo collections.
        run_task_locally(req)

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['calculation_type'] == 'slab+adsorbate optimization'
        assert fwork.name['adsorbate'] == adsorbate_name
        assert fwork.name['adsorbate_rotation'] == rotation
        assert isinstance(fwork.name['adsorbate_rotation'], dict)
        assert fwork.name['adsorption_site'] == adsorption_site
        assert fwork.name['mpid'] == mpid
        assert fwork.name['miller'] == miller_indices
        assert fwork.name['shift'] == shift
        assert fwork.name['top'] == top
        assert isinstance(fwork.name['slab_repeat'], tuple)
        assert fwork.name['dft_settings'] == ADSLAB_SETTINGS['vasp']
        assert task.complete() is True

    finally:
        clean_up_tasks()


def test__find_matching_adslab_doc():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make an Adslab from a bulk that shows up in the unit_testing_atoms Mongo
    collection. If you copy/paste this test into somewhere else, make sure
    that you use `run_task_locally` appropriately.
    '''
    # Make a case where this function should find something successfully
    task = GenerateAdslabs(adsorbate_name='CO', mpid='mp-2', miller_indices=(1, 0, 0))
    run_task_locally(task)
    docs = get_task_output(task)
    doc = MakeAdslabFW._find_matching_adslab_doc(docs,
                                                 adsorption_site=(1.48564485e-23,
                                                                  1.40646118e+00,
                                                                  2.08958465e+01),
                                                 shift=0.25, top=False)
    # I know what it should have found because I did this by hand
    expected_doc = docs[3]
    assert doc == expected_doc

    # Try a fail-to-find
    with pytest.raises(RuntimeError, message='Expected a RuntimeError') as exc_info:
        doc = MakeAdslabFW._find_matching_adslab_doc(docs, adsorption_site=(0., 0., 0.),
                                                     shift=0.25, top=False)
        assert ('You just tried to make an adslab FireWork rocket that we could not enumerate.'
                in str(exc_info.value))


def test_MakeSurfaceFW():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of actually
    submitting a FireWork to production. To avoid this, you must try to make a
    surface from a bulk that shows up in the unit_testing_atoms Mongo
    collection. If you copy/paste this test into somewhere else, make sure that
    you use `run_task_locally` appropriately.
    '''
    # Let's have our `FindSurface` task create an instance of `MakeSurfaceFW`
    # for us, because it's better at setting all the arguments and stuff
    mpid = 'mp-1018129'
    miller_indices = (0, 1, 1)
    shift = 0.5
    min_height = SLAB_SETTINGS['slab_generator_settings']['min_slab_size'],
    dft_settings = SLAB_SETTINGS['vasp']
    finder = FindSurface(mpid=mpid,
                         miller_indices=miller_indices,
                         shift=shift,
                         min_height=min_height,
                         dft_settings=dft_settings)
    try:
        schedule_tasks([finder.requires()], local_scheduler=True)
        finder._load_attributes()
        task = finder.dependency
        assert isinstance(task, MakeSurfaceFW)

        # Manually call the `run` method with the unit testing flag to get the
        # firework instead of actually submitting it
        fwork = task.run(_testing=True)
        assert fwork.name['calculation_type'] == 'surface energy optimization'
        assert fwork.name['mpid'] == mpid
        assert fwork.name['miller'] == miller_indices
        assert fwork.name['shift'] == shift
        assert fwork.name['num_slab_atoms'] == len(finder._create_surface())
        assert fwork.name['dft_settings'] == dft_settings
        assert task.complete() is True

    finally:
        clean_up_tasks()
