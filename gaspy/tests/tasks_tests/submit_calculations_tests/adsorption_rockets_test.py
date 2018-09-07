''' Tests for the `gaspy.tasks.submitters.adsorptions` submodule '''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ....tasks.rocket_builders.adsorption_rockets import (_standardize_miller,
                                                          _make_rocket_parameters_from_doc)

# Things we need to do the tests
import pytest
from collections import OrderedDict
from .... import defaults


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
def test__make_rocket_parameters_from_doc(doc, adsorbates, encut, xc, max_atoms):
    parameters = _make_rocket_parameters_from_doc(doc, adsorbates,
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
