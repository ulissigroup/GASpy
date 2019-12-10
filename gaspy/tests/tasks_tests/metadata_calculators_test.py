''' Tests for the `gaspy.tasks.metadata_calculators` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ...tasks.metadata_calculators import (CalculateRismAdsorptionEnergy,
                                           CalculateAdsorptionEnergy,
                                           CalculateAdsorbateEnergy,
                                           CalculateAtomicBasisEnergy,
                                           CalculateSurfaceEnergy)

# Things we need to do the tests
import pytest
import math
import numpy as np
import statsmodels.api as statsmodels
import ase
import pymatgen
from .utils import clean_up_tasks, run_task_locally
from ... import defaults
from ...mongo import make_atoms_from_doc
from ...utils import unfreeze_dict
from ...tasks import get_task_output, run_task
from ...tasks.core import schedule_tasks
from ...tasks.calculation_finders import FindBulk, FindSurface

GAS_SETTINGS = defaults.gas_settings()
SE_BULK_SETTINGS = defaults.surface_energy_bulk_settings()


def test_CalculateRismAdsorptionEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    adsorption_site = (2.558675812674303, 2.5586758126743008, 19.187941671)
    shift = 0.25
    top = True
    adsorbate_name = 'CO'
    mpid = 'mp-30'
    miller_indices = (1, 0, 0)
    task = CalculateRismAdsorptionEnergy(adsorption_site=adsorption_site,
                                         shift=shift,
                                         top=top,
                                         adsorbate_name=adsorbate_name,
                                         mpid=mpid,
                                         miller_indices=miller_indices,)

    try:
        run_task_locally(task)
        doc = get_task_output(task)

        # I just checked this one calculation by hand and found some key
        # information about it.
        assert math.isclose(doc['adsorption_energy'], -0.7436998067121294)
        assert doc['fwids']['slab'] == 3003
        assert doc['fwids']['adslab'] == 3006
        assert doc['fwids']['adsorbate'] == [3017, 3018, 3019]

    finally:
        clean_up_tasks()


def test_CalculateConstantMuAdsorptionEnergy():
    assert False


def test_CalculateAdsorptionEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    adsorption_site = (0., 1.41, 20.52)
    shift = 0.25
    top = True
    adsorbate_name = 'CO'
    mpid = 'mp-2'
    miller_indices = (1, 0, 0)
    task = CalculateAdsorptionEnergy(adsorption_site=adsorption_site,
                                     shift=shift,
                                     top=top,
                                     adsorbate_name=adsorbate_name,
                                     mpid=mpid,
                                     miller_indices=miller_indices,)

    try:
        run_task_locally(task)
        doc = get_task_output(task)

        # I just checked this one calculation by hand and found some key
        # information about it.
        assert math.isclose(doc['adsorption_energy'], -1.5959449799999899)
        assert doc['fwids']['slab'] == 124894
        assert doc['fwids']['adslab'] == 124897
        assert doc['fwids']['adsorbate'] == [19565, 19566, 19567]

    finally:
        clean_up_tasks()


def test_CalculateAdsorbateEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    adsorbate_name = 'OOH'
    dft_settings = GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    dft_settings=dft_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.dft_settings) == dft_settings

    try:
        run_task_locally(task)
        energy = get_task_output(task)
        assert energy == 2*(-7.19957549) + (-3.480310465)

    finally:
        clean_up_tasks()


def test_CalculateAdsorbateEnergy_Error():
    '''
    WARNING:  This test uses `run_task`, which has a chance of actually
    submitting a FireWork to production. To avoid this, you must try to make
    sure that you have all of the gas calculations in the unit testing atoms
    collection.  If you copy/paste this test into somewhere else, make sure
    that you use `run_task` appropriately.

    If we try to calculate the energy of an adsorbate that we have not yet
    defined, then this task should yell at us.
    '''
    adsorbate_name = 'U'
    dft_settings = GAS_SETTINGS['vasp']
    task = CalculateAdsorbateEnergy(adsorbate_name=adsorbate_name,
                                    dft_settings=dft_settings)
    assert task.adsorbate_name == adsorbate_name
    assert unfreeze_dict(task.dft_settings) == dft_settings

    try:
        with pytest.raises(KeyError, message='Expected a KeyError') as exc_info:
            run_task(task)
            assert ('You are trying to calculate the adsorbate energy of an '
                    'undefined adsorbate, U' in str(exc_info.value))

    finally:
        clean_up_tasks()


def test_CalculateAtomicBasisEnergy():
    '''
    WARNING:  This test uses `run_task_locally`, which has a chance of
    actually submitting a FireWork to production. To avoid this, you must try
    to make sure that you have all of the gas calculations in the unit testing
    atoms collection.  If you copy/paste this test into somewhere else, make
    sure that you use `run_task_locally` appropriately.
    '''
    expected_energies = {'H': -3.480310465,
                         'O': -7.19957549,
                         'C': -7.29110228,
                         'N': -8.08570028}

    dft_settings = GAS_SETTINGS['vasp']
    for atom, expected_energy in expected_energies.items():
        task = CalculateAtomicBasisEnergy(atom, dft_settings)
        assert unfreeze_dict(task.dft_settings) == dft_settings

        try:
            run_task_locally(task)
            energy = get_task_output(task)
            assert energy == expected_energy

        finally:
            clean_up_tasks()


class TestCalculateSurfaceEnergy():
    def test__static_requires(self):
        '''
        If the dependency isn't done, make sure that this method only returns
        it and doesn't do some of the other steps (i.e., assign attributes)
        '''
        mpid = 'mp-1018129'
        bulk_dft_settings = SE_BULK_SETTINGS['vasp']
        task = CalculateSurfaceEnergy(mpid=mpid, miller_indices=(0, 0, 1), shift=0.081)
        assert unfreeze_dict(task.bulk_dft_settings) == bulk_dft_settings
        bulk_task = task._static_requires()
        assert isinstance(bulk_task, FindBulk)
        assert bulk_task.mpid == mpid
        assert not hasattr(bulk_task, 'bulk_atoms')

        # If the dependency is done, make sure that the correct attributes are
        # there
        try:
            schedule_tasks([bulk_task], local_scheduler=True)
            bulk_task = task._static_requires()
            assert isinstance(bulk_task, FindBulk)
            assert bulk_task.mpid == mpid
            assert hasattr(task, 'bulk_atoms')
            assert hasattr(task, 'min_repeats')
            assert hasattr(task, 'unit_slab')
            assert hasattr(task, 'unit_slab_height')

        finally:
            clean_up_tasks()

    def test___terminate_if_too_large(self):
        '''
        Initialize the task and set `max_atoms` to something really low that'll
        hopefully fail. Also need to run the dependency manually for this to work.
        '''
        # Setup and pre-reqs
        mpid = 'mp-1018129'
        task = CalculateSurfaceEnergy(mpid=mpid,
                                      miller_indices=(0, 0, 1),
                                      shift=0.081,
                                      max_atoms=5)
        try:
            # Make sure it throws an error. Note that we don't actually call
            # the `__teminate_if_too_large` method, because it should be called
            # when we execute `_static_requires` with a completed `FindBulk`.
            with pytest.raises(RuntimeError, match='Cannot calculate surface.*'):
                bulk_task = task._static_requires()
                schedule_tasks([bulk_task], local_scheduler=True)
                _ = task._static_requires()  # noqa: F841

        finally:
            clean_up_tasks()

    def test___calculate_unit_slab(self):
        '''
        This is a pretty lazy test that just checks that the method creates the
        correct type of objects. It doesn't check if it's done correctly
        though. Feel free to fix that if you want.
        '''
        # This initialization should implicitly call the
        # `__calculate_unit_slab` method
        mpid = 'mp-1018129'
        task = CalculateSurfaceEnergy(mpid=mpid, miller_indices=(0, 0, 1), shift=0.081)
        bulk_task = task._static_requires()
        try:
            schedule_tasks([bulk_task], local_scheduler=True)
            bulk_task = task._static_requires()

            # Are the right attributes set?
            assert isinstance(task.unit_slab, pymatgen.core.surface.Slab)
            assert isinstance(task.unit_slab_height, float)

        finally:
            clean_up_tasks()

    def test__dynamic_requires(self):
        '''
        Run the static requirements first (which feed the dynamic ones), then
        check if the dynamic requirements were configured correctly.
        '''
        # Setup
        mpid = 'mp-1018129'
        miller = (0, 0, 1)
        shift = 0.081
        task = CalculateSurfaceEnergy(mpid=mpid, miller_indices=miller, shift=shift)
        bulk_task = task._static_requires()
        try:
            schedule_tasks([bulk_task], local_scheduler=True)
            bulk_task = task._static_requires()
            dynam_reqs = task._dynamic_requires()

            # Test
            assert dynam_reqs == task.surface_relaxation_tasks
            assert len(dynam_reqs) == 3
            for i, req in enumerate(dynam_reqs):
                assert isinstance(req, FindSurface)
                assert req.mpid == mpid
                assert req.miller_indices == miller
                assert req.shift == shift
                expected_min_height = task.unit_slab_height * (i+task.min_repeats)
                assert req.min_height == expected_min_height

        finally:
            clean_up_tasks()

    def test_run(self):
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=0.081)
        try:
            schedule_tasks([task], local_scheduler=True)
            doc = get_task_output(task)

            assert all(isinstance(make_atoms_from_doc(doc_), ase.Atoms)
                       for doc_ in doc['surface_structures'])
            assert 'surface_energy' in doc
            assert 'surface_energy_standard_error' in doc

        finally:
            clean_up_tasks()

    def test__calculate_surface_energy(self):
        '''
        Yeah this test is just a copy/paste of the original function. I'm too
        lazy to figure out how to do it better.
        '''
        # Get the calculated surface energy and uncertainty
        task = CalculateSurfaceEnergy(mpid='mp-1018129', miller_indices=(0, 0, 1), shift=0.081)
        try:
            schedule_tasks([task], local_scheduler=True)
            doc = get_task_output(task)
            surface_energy = doc['surface_energy']
            surface_energy_se = doc['surface_energy_standard_error']

            # Recalculate the surface energy ourselves
            atoms_list = [make_atoms_from_doc(doc_) for doc_ in doc['surface_structures']]
            n_atoms = [len(atoms) for atoms in atoms_list]
            slab_energies = [atoms.get_potential_energy() for atoms in atoms_list]
            area = 2 * np.linalg.norm(np.cross(atoms_list[0].cell[0], atoms_list[0].cell[1]))
            slab_energies_per_area = slab_energies/area
            data = statsmodels.add_constant(n_atoms)
            mod = statsmodels.OLS(slab_energies_per_area, data)
            res = mod.fit()
            expected_surface_energy = res.params[0]
            expected_surface_energy_se = res.bse[0]

            # Compare
            assert math.isclose(surface_energy, expected_surface_energy)
            assert math.isclose(surface_energy_se, expected_surface_energy_se)

        finally:
            clean_up_tasks()
