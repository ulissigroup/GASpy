''' Tests for the `gaspy.tasks.calculation_finders` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.calculation_finders import (FindCalculation,
                                          FindGas,
                                          FindBulk,
                                          FindAdslab,
                                          calculate_surface_k_points,
                                          FindRismAdslab,
                                          FindSurface)

# Things we need to do the tests
import os
import pytest
import warnings
import math
import numpy as np
import luigi
import ase
from .utils import clean_up_tasks
from ... import defaults
from ...utils import unfreeze_dict
from ...mongo import make_atoms_from_doc
from ...tasks.core import get_task_output, schedule_tasks
from ...tasks.make_fireworks import (MakeGasFW,
                                     MakeBulkFW,
                                     MakeAdslabFW,
                                     MakeRismAdslabFW,
                                     MakeSurfaceFW)

GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


def test_FindCalculation():
    '''
    We do a very light test of this parent class, because we will rely more
    heavily on the testing on the child classes and methods.
    '''
    finder = FindCalculation()
    assert isinstance(finder, luigi.Task)
    assert hasattr(finder, 'run')
    assert hasattr(finder, 'output')
    assert hasattr(finder, 'max_fizzles')

    # Ok, let's pick one of the child tasks to test the max_fizzles feature.
    mpid = 'mp-42'
    dft_settings = BULK_SETTINGS['vasp']
    task = FindBulk(mpid=mpid, dft_settings=dft_settings, max_fizzles=0)
    try:
        with pytest.raises(ValueError, match='Since we have fizzled'):
            _ = list(task.run(_testing=True))     # noqa: F841
    finally:
        clean_up_tasks()


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

    # If there's nothing, make sure we get nothing
    docs = []
    assert remove_old_docs(docs) == {}


def _assert_dft_settings(doc, dft_settings):
    '''
    Asserts whether the dft settings inside a doc/dictionary are correct based
    on whether they look like VASP settings or Quantum Espresso settings.

    Args:
        doc             Dictionary/Mongo document object
        dft_settings    Dictionary of VASP settings
    '''
    if dft_settings['_calculator'] == 'vasp':
        _assert_vasp_settings(doc, dft_settings)
    elif dft_settings['_calculator'] == 'qe':
        _assert_qe_settings(doc, dft_settings)
    elif dft_settings['_calculator'] == 'rism':
        _assert_qe_settings(doc, dft_settings)
    else:
        raise AssertionError('The DFT settings do not look like anything we '
                             'have the infrastruture setup for.')


def _assert_vasp_settings(doc, vasp_settings):
    '''
    Asserts whether the vasp_settings inside a doc/dictionary are correct

    Args:
        doc             Dictionary/Mongo document object
        vasp_settings   Dictionary of VASP settings
    '''
    for key, value in vasp_settings.items():
        try:
            assert doc['fwname']['dft_settings'][key] == value

        # Some of our VASP settings are tuples, but Mongo only saves lists.
        # If we're looking at one of these cases, then we should compare
        # list-to-list
        except AssertionError:
            if isinstance(value, tuple):
                assert doc['fwname']['dft_settings'][key] == list(value)

        except KeyError:
            # If we're looking at an adslab, then we don't care about certain
            # vasp settings
            if (doc['fwname']['calculation_type'] == 'slab+adsorbate optimization' and
                key in {'nsw', 'isym', 'symprec'}):  # noqa: E129
                pass

            # If we're looking at a slab, then we don't care about certain
            # vasp settings
            elif (doc['fwname']['calculation_type'] == 'slab+adsorbate optimization' and
                  key in {'isym'}):
                pass

            else:
                raise


def _assert_qe_settings(doc, rism_settings):
    '''
    Asserts whether the qe_settings inside a doc/dictionary are correct

    Args:
        doc             Dictionary/Mongo document object
        rism_settings   Dictionary of RISM settings
    '''
    for key, value in rism_settings.items():
        try:
            assert doc['fwname']['dft_settings'][key] == value

        # The kpts key is calculated on-the-fly. So we will input 'surface' and
        # get out a 3-tuple of integers. This is ok.
        except AssertionError:
            if key == 'kpts':
                kpts = doc['fwname']['dft_settings'][key]
                assert isinstance(kpts, tuple) or isinstance(kpts, list)
                assert len(kpts) == 3
                for kpt in kpts:
                    assert isinstance(kpt, int) or isinstance(kpt, float)
            else:
                raise


def test_FindGas_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    gas = 'H2'
    dft_settings = GAS_SETTINGS['vasp']
    task = FindGas(gas_name=gas, dft_settings=dft_settings)

    try:
        _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)
        assert doc['fwname']['calculation_type'] == 'gas phase optimization'
        assert doc['fwname']['gasname'] == gas
        _assert_dft_settings(doc, dft_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_tasks()


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
    task = FindGas(gas_name=gas, dft_settings=GAS_SETTINGS['vasp'])

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeGasFW)
        assert dependency.gas_name == gas

    finally:
        clean_up_tasks()


def test_FindBulk_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    mpid = 'mp-2'
    dft_settings = BULK_SETTINGS['vasp']
    task = FindBulk(mpid=mpid, dft_settings=dft_settings)

    try:
        _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)

        assert doc['fwname']['calculation_type'] == 'unit cell optimization'
        assert doc['fwname']['mpid'] == mpid
        _assert_dft_settings(doc, dft_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_tasks()


def test_FindBulk_unsuccessfully():
    '''
    If we ask this task to find something that is not there, it should return
    the correct dependency
    '''
    mpid = 'mp-42'
    task = FindBulk(mpid=mpid, dft_settings=BULK_SETTINGS['vasp'])

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeBulkFW)
        assert dependency.mpid == mpid

    finally:
        clean_up_tasks()


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
    dft_settings = ADSLAB_SETTINGS['vasp']
    task = FindAdslab(adsorption_site=adsorption_site,
                      shift=shift,
                      top=top,
                      adsorbate_name=adsorbate_name,
                      rotation=rotation,
                      mpid=mpid,
                      miller_indices=miller_indices,
                      dft_settings=dft_settings)

    try:
        _run_task_with_dynamic_dependencies(task)
        doc = get_task_output(task)
        assert doc['fwname']['calculation_type'] == 'slab+adsorbate optimization'
        assert doc['fwname']['adsorption_site'] == list(adsorption_site)
        assert math.isclose(doc['fwname']['shift'], shift)
        assert doc['fwname']['top'] == top
        assert doc['fwname']['adsorbate'] == adsorbate_name
        assert doc['fwname']['adsorbate_rotation'] == rotation
        assert doc['fwname']['mpid'] == mpid
        assert tuple(doc['fwname']['miller']) == miller_indices
        _assert_dft_settings(doc, dft_settings)

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_tasks()


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
    dft_settings = ADSLAB_SETTINGS['vasp']
    task = FindAdslab(adsorption_site=adsorption_site,
                      shift=shift,
                      top=top,
                      adsorbate_name=adsorbate_name,
                      rotation=rotation,
                      mpid=mpid,
                      miller_indices=miller_indices,
                      dft_settings=dft_settings)

    try:
        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeAdslabFW)
        assert dependency.mpid == mpid
        assert dependency.adsorption_site == adsorption_site
        assert dependency.shift == shift
        assert dependency.top == top
        assert dependency.adsorbate_name == adsorbate_name
        assert unfreeze_dict(dependency.rotation) == rotation
        assert dependency.miller_indices == miller_indices
        assert unfreeze_dict(dependency.dft_settings) == dft_settings

    finally:
        clean_up_tasks()


def test_calculate_surface_k_points():
    for cell in [[[9.9, 0.0, 0.0],
                  [0.0, 10.3, 0.0],
                  [0.0, -6.0, 20.1]],
                 [[5.53, 0.00, 0.00],
                  [-1.84, 13.79, 0.00],
                  [7.37, 3.94, 20.48]]]:
        atoms = ase.Atoms('C', cell=cell)
        k_points = calculate_surface_k_points(atoms)

        a0, b0, c0 = (np.linalg.norm(vector) for vector in cell)
        assert all(isinstance(point, int) for point in k_points)
        assert k_points[0] == int(20 / a0)
        assert k_points[1] == max(1, int(k_points[0] * a0 / b0))
        assert k_points[2] == 1


class TestFindSurface():
    def test__create_surface(self):
        '''
        This is a very bad test, because it really only does a type check. We
        should probably verify that the height is correct and that the surface
        has the correct orientation. Thanks for fixing this, future programmer!
        '''
        mpid = 'mp-1018129'
        miller_indices = (0, 1, 1)
        shift = 0.5
        min_height = SLAB_SETTINGS['slab_generator_settings']['min_slab_size'],
        dft_settings = SLAB_SETTINGS['vasp']
        task = FindSurface(mpid=mpid,
                           miller_indices=miller_indices,
                           shift=shift,
                           min_height=min_height,
                           dft_settings=dft_settings)
        try:
            schedule_tasks([task.requires()], local_scheduler=True)

            # Should be more than just checking type
            surface = task._create_surface()
            assert isinstance(surface, ase.Atoms)

        finally:
            clean_up_tasks()

    def test__constrain_slab(self):
        '''
        Test this static method on all our test slabs
        '''
        # Pull the static method out
        constrain_slab = FindSurface._FindSurface__constrain_surface

        # Run the method for each slab
        z_cutoff = 3.
        test_dir = '/home/GASpy/gaspy/tests/test_cases/slabs/'
        for file_name in os.listdir(test_dir):
            atoms = ase.io.read(os.path.join(test_dir, file_name))
            atoms.set_constraint()  # Clear out any constraints that might be there already
            constrained_atoms = constrain_slab(atoms, z_cutoff=z_cutoff)

            # Verify that the correct atoms are constrained
            z_positions = [atom.position[2] for atom in atoms]
            upper_cutoff = max(z_positions) - z_cutoff
            lower_cutoff = min(z_positions) + z_cutoff
            for i, atom in enumerate(constrained_atoms):
                if lower_cutoff < atom.position[2] < upper_cutoff:
                    assert i in constrained_atoms.constraints[0].index
                else:
                    assert i not in constrained_atoms.constraints[0].index

    def test_successful_find(self):
        '''
        If we ask this task to find something that is there, it should return
        the correct Mongo document/dictionary
        '''
        mpid = 'mp-1018129'
        miller_indices = (0, 1, 1)
        shift = 0.5
        min_height = SLAB_SETTINGS['slab_generator_settings']['min_slab_size'],
        dft_settings = SLAB_SETTINGS['vasp']
        task = FindSurface(mpid=mpid,
                           miller_indices=miller_indices,
                           shift=shift,
                           min_height=min_height,
                           dft_settings=dft_settings)

        try:
            _run_task_with_dynamic_dependencies(task.requires())
            _run_task_with_dynamic_dependencies(task)
            doc = get_task_output(task)
            assert doc['fwname']['calculation_type'] == 'surface energy optimization'
            assert doc['fwname']['mpid'] == mpid
            assert math.isclose(doc['fwname']['shift'], shift)
            assert tuple(doc['fwname']['miller']) == miller_indices
            assert task.min_height == min_height
            _assert_dft_settings(doc, dft_settings)

            # Make sure we can turn it into an atoms object
            assert isinstance(make_atoms_from_doc(doc), ase.Atoms)

        finally:
            clean_up_tasks()

    def test_unsuccessful_find(self):
        '''
        If we ask this task to find something that is not there, it should return
        the correct dependency
        '''
        mpid = 'mp-1018129'
        miller_indices = (0, 1, 1)
        shift = 90001.
        min_height = SLAB_SETTINGS['slab_generator_settings']['min_slab_size'],
        dft_settings = SLAB_SETTINGS['vasp']
        task = FindSurface(mpid=mpid,
                           miller_indices=miller_indices,
                           shift=shift,
                           min_height=min_height,
                           dft_settings=dft_settings)

        try:
            _run_task_with_dynamic_dependencies(task.requires())
            dependency = _run_task_with_dynamic_dependencies(task)
            assert isinstance(dependency, MakeSurfaceFW)
            assert dependency.mpid == mpid
            assert dependency.miller_indices == miller_indices
            assert dependency.shift == shift
            # Test all the DFT settings, excluding the kpts (which are
            # calculated on the spot).
            for key, value in dft_settings.items():
                try:
                    assert dependency.dft_settings[key] == value
                except AssertionError:
                    if key == 'kpts':
                        pass
                    else:
                        raise

        finally:
            clean_up_tasks()


def test_RismFindAdslab_successfully():
    '''
    If we ask this task to find something that is there, it should return
    the correct Mongo document/dictionary
    '''
    # Various settings
    adsorption_site = (2.558675812674303, 2.5586758126743008, 19.187941671)
    shift = 0.25
    top = True
    adsorbate_name = 'CO'
    rotation = {'phi': 0., 'theta': 0., 'psi': 0.}
    mpid = 'mp-30'
    miller_indices = (1, 0, 0)

    # Need to run QE before running RISM
    req = FindAdslab(adsorption_site=adsorption_site,
                     shift=shift,
                     top=top,
                     adsorbate_name=adsorbate_name,
                     rotation=rotation,
                     mpid=mpid,
                     miller_indices=miller_indices,
                     dft_settings=ADSLAB_SETTINGS['qe'])
    try:
        schedule_tasks([req])
        qe_doc = get_task_output(req)
        for key in ['ctime', 'mtime', '_id', 'calculation_date', 'initial_configuration']:
            del qe_doc[key]

        # Need to feed the positions from QE relaxation to RISM
        task = FindRismAdslab(atoms_dict=qe_doc,
                              adsorption_site=adsorption_site,
                              shift=shift,
                              top=top,
                              adsorbate_name=adsorbate_name,
                              rotation=rotation,
                              mpid=mpid,
                              miller_indices=miller_indices,
                              dft_settings=ADSLAB_SETTINGS['rism'])
        schedule_tasks([task])
        doc = get_task_output(task)
        assert doc['fwname']['calculation_type'] == 'slab+adsorbate optimization'
        assert doc['fwname']['adsorption_site'] == list(adsorption_site)
        assert math.isclose(doc['fwname']['shift'], shift)
        assert doc['fwname']['top'] == top
        assert doc['fwname']['adsorbate'] == adsorbate_name
        assert doc['fwname']['adsorbate_rotation'] == rotation
        assert doc['fwname']['mpid'] == mpid
        assert tuple(doc['fwname']['miller']) == miller_indices
        _assert_dft_settings(doc, ADSLAB_SETTINGS['rism'])

        # Make sure we can turn it into an atoms object
        _ = make_atoms_from_doc(doc)    # noqa: F841

    finally:
        clean_up_tasks()


def test_RismFindAdslab_unsuccessfully():
    '''
    If we ask this task to find something that is not there, it should return
    the correct dependency
    '''
    # Various settings
    adsorption_site = (2.558675812674303, 2.5586758126743008, 19.187941671)
    shift = 0.25
    top = True
    adsorbate_name = 'OH'
    rotation = {'phi': 0., 'theta': 0., 'psi': 0.}
    mpid = 'mp-30'
    miller_indices = (1, 0, 0)

    # Need to run QE before running RISM
    req = FindAdslab(adsorption_site=adsorption_site,
                     shift=shift,
                     top=top,
                     adsorbate_name='CO',
                     rotation=rotation,
                     mpid=mpid,
                     miller_indices=miller_indices,
                     dft_settings=ADSLAB_SETTINGS['qe'])
    try:
        schedule_tasks([req])
        qe_doc = get_task_output(req)
        for key in ['ctime', 'mtime', '_id', 'calculation_date', 'initial_configuration']:
            del qe_doc[key]
        task = FindRismAdslab(atoms_dict=qe_doc,
                              adsorption_site=adsorption_site,
                              shift=shift,
                              top=top,
                              adsorbate_name=adsorbate_name,
                              rotation=rotation,
                              mpid=mpid,
                              miller_indices=miller_indices,
                              dft_settings=ADSLAB_SETTINGS['rism'])

        dependency = _run_task_with_dynamic_dependencies(task)
        assert isinstance(dependency, MakeRismAdslabFW)
        assert dependency.mpid == mpid
        assert dependency.adsorption_site == adsorption_site
        assert dependency.shift == shift
        assert dependency.top == top
        assert dependency.adsorbate_name == adsorbate_name
        assert unfreeze_dict(dependency.rotation) == rotation
        assert dependency.miller_indices == miller_indices
        assert unfreeze_dict(dependency.dft_settings) == ADSLAB_SETTINGS['rism']

    finally:
        clean_up_tasks()
