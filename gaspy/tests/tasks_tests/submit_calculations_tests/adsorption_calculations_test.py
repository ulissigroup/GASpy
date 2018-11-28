''' Tests for the `gaspy.tasks.submitters.adsorptions` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Modify the python path so that we find/use the .gaspyrc.json in the testing
# folder instead of the main folder
import os
os.environ['PYTHONPATH'] = '/home/GASpy/gaspy/tests:' + os.environ['PYTHONPATH']

# Things we're testing
from ....tasks.submit_calculations.adsorption_calculations import (_standardize_miller,
                                                                   _make_adslab_parameters_from_doc,
                                                                   _make_relaxation_tasks_from_parameters)

# Things we need to do the tests
import pytest
from collections import OrderedDict
from .... import defaults
from ....tasks import FingerprintRelaxedAdslab
from ....gasdb import get_unsimulated_catalog_docs


@pytest.mark.parametrize('miller',
                         [[1, 1, 1],
                          '[1, 1, 1]',
                          '[1,1,1]',
                          '[1,1, 1]',
                          (1, 1, 1),
                          '(1, 1, 1)',
                          '(1, 1,1)'])
def test__standardize_miller(miller):
    standardized_miller = _standardize_miller(miller)
    assert standardized_miller == '[1,1,1]'


@pytest.mark.parametrize('doc,adsorbates,encut,bulk_encut,slab_encut,xc,pp_version,max_bulk_atoms',
                         [({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': True,
                            'adsorption_site': [0., 0., 0.],
                            'adsorbate_rotation': {'phi': 0., 'theta': 0., 'psi': 0.}},
                           ['CO'], 350., 350., 350., 'rpbe', '5.4', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': True,
                            'adsorption_site': [0., 0., 0.],
                            'adsorbate_rotation': {'phi': 0., 'theta': 0., 'psi': 0.}},
                           ['CO'], 350., 500., 350., 'rpbe', '5.4', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False,
                            'adsorption_site': [0., 0., 0.],
                            'adsorbate_rotation': {'phi': 0., 'theta': 0., 'psi': 0.}},
                           ['CO'], 350., 500., 350., 'rpbe', '5.4', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False,
                            'adsorption_site': [1., 1., 1.],
                            'adsorbate_rotation': {'phi': 0., 'theta': 0., 'psi': 0.}},
                           ['CO'], 350., 500., 350., 'rpbe', '5.4', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False,
                            'adsorption_site': [1., 1., 1.],
                            'adsorbate_rotation': {'phi': 0., 'theta': 0., 'psi': 0.}},
                           ['H'], 350., 500., 350., 'rpbe', '5.4', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False,
                            'adsorption_site': [1., 1., 1.]},
                           ['H'], 350., 500., 350., 'rpbe', '5.4', 80)])
def test__make_adslab_parameters_from_doc(doc, adsorbates, encut, bulk_encut, slab_encut,
                                          xc, pp_version, max_bulk_atoms):
    parameters = _make_adslab_parameters_from_doc(doc, adsorbates,
                                                  encut=encut,
                                                  bulk_encut=bulk_encut,
                                                  slab_encut=slab_encut,
                                                  xc=xc,
                                                  pp_version=pp_version,
                                                  max_bulk_atoms=max_bulk_atoms)

    # Deal with cases where we the document doesn't have a specified rotation
    try:
        adsorbate_rotation = doc['adsorbate_rotation']
    except KeyError:
        adsorbate_rotation = defaults.ROTATION

    expected_parameters = OrderedDict.fromkeys(['bulk', 'slab', 'adsorption', 'gas'])
    expected_parameters['bulk'] = defaults.bulk_parameters(mpid=doc['mpid'],
                                                           settings=xc,
                                                           encut=bulk_encut,
                                                           pp_version=pp_version,
                                                           max_atoms=max_bulk_atoms)
    expected_parameters['slab'] = defaults.slab_parameters(miller=doc['miller'],
                                                           top=doc['top'],
                                                           shift=doc['shift'],
                                                           settings=xc,
                                                           encut=slab_encut,
                                                           pp_version=pp_version)
    expected_parameters['adsorption'] = defaults.adsorption_parameters(adsorbate=adsorbates[0],
                                                                       adsorption_site=doc['adsorption_site'],
                                                                       adsorbate_rotation=adsorbate_rotation,
                                                                       settings=xc,
                                                                       encut=encut,
                                                                       pp_version=pp_version)
    expected_parameters['gas'] = defaults.gas_parameters(adsorbates[0],
                                                         settings=xc,
                                                         encut=encut,
                                                         pp_version=pp_version)
    assert parameters == expected_parameters


@pytest.mark.parametrize('adsorbates', [['CO'], ['H']])
def test__make_relaxation_tasks_from_parameters(adsorbates):
    docs = get_unsimulated_catalog_docs(adsorbates)
    parameters_list = [_make_adslab_parameters_from_doc(doc, adsorbates=['CO']) for doc in docs]
    tasks = _make_relaxation_tasks_from_parameters(parameters_list)

    for task in tasks:
        assert isinstance(task, FingerprintRelaxedAdslab)
