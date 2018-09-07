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
from ....gasdb import get_catalog_docs


@pytest.mark.parametrize('miller',
                         [[1, 1, 1],
                          '[1, 1, 1]',
                          '[1,1,1]',
                          '[1,1, 1]'])
def test__standardize_miller(miller):
    standardized_miller = _standardize_miller(miller)
    assert isinstance(standardized_miller, str)
    assert ' ' not in standardized_miller


@pytest.mark.parametrize('doc,adsorbates,encut,xc,max_atoms',
                         [({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': True, 'adsorption_site': [0., 0., 0.]}, ['CO'], 500., 'rpbe', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': True, 'adsorption_site': [0., 0., 0.]}, ['CO'], 350., 'rpbe', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False, 'adsorption_site': [0., 0., 0.]}, ['CO'], 350., 'rpbe', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False, 'adsorption_site': [1., 1., 1.]}, ['CO'], 350., 'rpbe', 80),
                          ({'mpid': 'mp-30.', 'miller': [1, 1, 1], 'shift': 0., 'top': False, 'adsorption_site': [1., 1., 1.]}, ['H'], 350., 'rpbe', 80)])
def test__make_adslab_parameters_from_doc(doc, adsorbates, encut, xc, max_atoms):
    parameters = _make_adslab_parameters_from_doc(doc, adsorbates,
                                                  encut=encut, xc=xc,
                                                  max_atoms=max_atoms)

    expected_parameters = OrderedDict.fromkeys(['bulk', 'slab', 'adsorption', 'gas'])
    expected_parameters['bulk'] = defaults.bulk_parameters(mpid=doc['mpid'],
                                                           encut=encut,
                                                           settings=xc,
                                                           max_atoms=max_atoms)
    expected_parameters['slab'] = defaults.slab_parameters(miller=doc['miller'],
                                                           top=doc['top'],
                                                           shift=doc['shift'],
                                                           settings=xc)
    expected_parameters['adsorption'] = defaults.adsorption_parameters(adsorbate=adsorbates[0],
                                                                       adsorption_site=doc['adsorption_site'],
                                                                       settings=xc)
    expected_parameters['gas'] = defaults.gas_parameters(adsorbates[0], settings=xc)
    assert parameters == expected_parameters


def test__make_relaxation_tasks_from_parameters():
    docs = get_catalog_docs()
    parameters_list = [_make_adslab_parameters_from_doc(doc, adsorbates=['CO']) for doc in docs]
    tasks = _make_relaxation_tasks_from_parameters(parameters_list)

    for task in tasks:
        assert isinstance(task, FingerprintRelaxedAdslab)
