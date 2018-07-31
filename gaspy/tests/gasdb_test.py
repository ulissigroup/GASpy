'''
Tests for the `gasdb` submodule. Since this submodule deals with Mongo databases so much,
we created and work with various collections that are tagged with `unit_testing_adsorption`.
We assume that these collections have very specific contents, as outlined in the
`get_expected_*_documents` functions in the `gaspy.tests.baselines` module.
We also assume that documents within each collection should have the same exact JSON architecture.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# Things we're testing
from ..gasdb import get_adsorption_docs, \
    _clean_up_aggregated_docs, \
    get_catalog_docs, \
    get_surface_docs, \
    hash_docs, \
    hash_doc

# Things we need to do the tests
import copy
import datetime
from bson.objectid import ObjectId
import random
from .baselines import get_expected_adsorption_documents


def test_get_adsorption_docs():
    expected_docs = get_expected_aggregated_adsorption_documents()
    docs = get_adsorption_docs(_collection_tag='unit_testing_adsorption')
    assert docs == expected_docs


def get_expected_aggregated_adsorption_documents():
    '''
    Note that we have only one document because the other one should have been filtered out.
    '''
    docs = [{'adslab_calculation_date': datetime.datetime(2017, 5, 16, 8, 56, 29, 993000),
             'adsorbates': ['H'],
             'coordination': 'Al-Ni',
             'energy': -0.16585670499999816,
             'formula': 'HAl6Ni10',
             'miller': [0, 0, 1],
             'mongo_id': ObjectId('59a015cbd3952577173b122d'),
             'mpid': 'mp-16514',
             'neighborcoord': ['Al:Ni-Ni-Ni-Ni-Ni-Ni-Ni', 'Ni:Al-Al-Al-Al-Al-Ni-Ni'],
             'nextnearestcoordination': 'Al-Al-Al-Al-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni',
             'shift': 0,
             'top': True}]
    return docs


def test__clean_up_aggregated_docs():
    expected_docs = get_expected_adsorption_documents()
    dirty_docs = __make_documents_dirty(expected_docs)
    clean_docs = _clean_up_aggregated_docs(dirty_docs, expected_keys=expected_docs[0].keys())
    assert _compare_unordered_sequences(clean_docs, expected_docs)


def __make_documents_dirty(docs):
    ''' Helper function for `test__clean_up_docs`  '''
    # Make a document with a missing key/value item
    doc_partial = copy.deepcopy(random.choice(docs))
    key_to_delete = random.choice(list(doc_partial.keys()))
    _ = doc_partial.pop(key_to_delete)   # noqa: F841

    # Make a document with a `None` value
    doc_empty = copy.deepcopy(random.choice(docs))
    key_to_modify = random.choice(list(doc_empty.keys()))
    doc_empty[key_to_modify] = None

    # Make the dirty documents
    dirty_docs = copy.deepcopy(docs) + [doc_partial] + [doc_empty]
    dirty_docs = [{'_id': doc} for doc in dirty_docs]   # because mongo is stupid
    return dirty_docs


def _compare_unordered_sequences(seq0, seq1):
    '''
    If (1) we want to see if two sequences are identical, (2) do not care about ordering,
    (3) the items are not hashable, and (4) items are not orderable, then use this
    function to compare them. Credit goes to CrowbarKZ on StackOverflow.
    '''
    seq0 = list(seq0)   # make a mutable copy
    try:
        for element in seq1:
            seq0.remove(element)
    except ValueError:
        return False
    return not seq0


def test_get_catalog_docs():
    expected_docs = get_expected_aggregated_catalog_documents()
    docs = get_catalog_docs(_collection_tag='unit_testing_catalog')
    assert docs == expected_docs


def get_expected_aggregated_catalog_documents():
    docs = [{'coordination': 'Pd-Pd',
             'formula': 'Pd16U',
             'miller': [1, 0, 0],
             'mongo_id': ObjectId('597b9bea899e208675296e00'),
             'mpid': 'mp-2',
             'neighborcoord': ['Pd:Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd', 'Pd:Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd'],
             'nextnearestcoordination': 'Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd',
             'shift': 0.25,
             'top': True},
            {'coordination': 'Sb-Sb',
             'formula': 'In2Sb2U',
             'miller': [0, 0, 1],
             'mongo_id': ObjectId('597e076b0ea601ba636e3db0'),
             'mpid': 'mp-1007661',
             'neighborcoord': ['Sb:In', 'Sb:In'],
             'nextnearestcoordination': 'Sb',
             'shift': 0.425,
             'top': True},
            {'coordination': 'Pd',
             'formula': 'Pd16U',
             'miller': [1, 0, 0],
             'mongo_id': ObjectId('597b9bea899e208675296dff'),
             'mpid': 'mp-2',
             'neighborcoord': ['Pd:Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd'],
             'nextnearestcoordination': 'Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd-Pd',
             'shift': 0.25,
             'top': True}]
    return docs


def test_get_surface_docs():
    expected_docs = get_expected_aggregated_surface_documents()
    docs = get_surface_docs(_collection_tag='unit_testing_surface_energy')
    assert docs == expected_docs


def get_expected_aggregated_surface_documents():
    docs = [{'FW_info': {'4': 233253, '6': 233256, '8': 233254, 'bulk': 65223},
             'formula': 'Ag4',
             'initial_configuration': {'atoms': {'atoms': [{'charge': 0.0,
                                                            'index': 0,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         15.789038563812309],
                                                            'symbol': 'Ag',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 1,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.48860482,
                                                                         1.48860482,
                                                                         17.894243646708205],
                                                            'symbol': 'Ag',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 2,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         11.578628103291795],
                                                            'symbol': 'Ag',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 3,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.48860482,
                                                                         1.48860482,
                                                                         13.683833186187693],
                                                            'symbol': 'Ag',
                                                            'tag': 0}],
                                                 'cell': [[2.97720964, -0.0, 0.0],
                                                          [-0.0, 2.97720964, 0.0],
                                                          [0.0, 0.0, 29.47287175]],
                                                 'chemical_symbols': ['Ag'],
                                                 'constraints': [{'kwargs': {'indices': []},
                                                                  'name': 'FixAtoms'}],
                                                 'info': {},
                                                 'mass': 431.4728,
                                                 'natoms': 4,
                                                 'pbc': [True, True, True],
                                                 'spacegroup': 'P4/nmm (129)',
                                                 'symbol_counts': {'Ag': 4},
                                                 'volume': 261.2409698300885},
                                       'calculator': {'class': 'SinglePointCalculator',
                                                      'module': 'ase.calculators.singlepoint'},
                                       'ctime': datetime.datetime(2018, 4, 13, 23, 27, 30, 658000),
                                       'inserted-hash': '8745d2aff9f3cd0d0993fdfbae19b47b7e647372',
                                       'mtime': datetime.datetime(2018, 4, 13, 23, 27, 30, 658000),
                                       'results': {'energy': -8.31506113,
                                                   'fmax': 0.38448162,
                                                   'fmax_unconstrained': 0.38448162,
                                                   'forces': [[-1.02e-06,
                                                               -2.66e-06,
                                                               0.19455533],
                                                              [-2.09e-06,
                                                               2.9e-06,
                                                               -0.3840541],
                                                              [2.78e-06,
                                                               2.61e-06,
                                                               0.38448162],
                                                              [3.3e-07,
                                                               -2.86e-06,
                                                               -0.19498285]]},
                                       'user': 'zulissi'},
             'intercept': 0.03264778440175653,
             'intercept_uncertainty': 0.005675090127581693,
             'miller': [1, 0, 0],
             'mongo_id': ObjectId('5ad4ca5c0ea601a67243eb9a'),
             'mpid': 'mp-124'},
            {'FW_info': {'4': 233246, '6': 233245, '8': 233248, 'bulk': 59076},
             'formula': 'Au4',
             'initial_configuration': {'atoms': {'atoms': [{'charge': 0.0,
                                                            'index': 0,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         15.735325711596891],
                                                            'symbol': 'Au',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 1,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.483540725,
                                                                         1.483540725,
                                                                         17.833369081064593],
                                                            'symbol': 'Au',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 2,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         11.539238678935407],
                                                            'symbol': 'Au',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 3,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.483540725,
                                                                         1.483540725,
                                                                         13.63728204840311],
                                                            'symbol': 'Au',
                                                            'tag': 0}],
                                                 'cell': [[2.96708145, 0.0, 0.0],
                                                          [0.0, 2.96708145, 0.0],
                                                          [0.0, 0.0, 29.37260776]],
                                                 'chemical_symbols': ['Au'],
                                                 'constraints': [{'kwargs': {'indices': []},
                                                                  'name': 'FixAtoms'}],
                                                 'info': {},
                                                 'mass': 787.866276,
                                                 'natoms': 4,
                                                 'pbc': [True, True, True],
                                                 'spacegroup': 'P4/nmm (129)',
                                                 'symbol_counts': {'Au': 4},
                                                 'volume': 258.5838769633163},
                                       'calculator': {'class': 'SinglePointCalculator',
                                                      'module': 'ase.calculators.singlepoint'},
                                       'ctime': datetime.datetime(2018, 4, 13, 23, 27, 30, 461000),
                                       'inserted-hash': 'f550a05b27b42322b03ca2f5791c370c9f136dfa',
                                       'mtime': datetime.datetime(2018, 4, 13, 23, 27, 30, 461000),
                                       'results': {'energy': -10.03514039,
                                                   'fmax': 0.48540866,
                                                   'fmax_unconstrained': 0.48540866,
                                                   'forces': [[-1.125e-05,
                                                               2.27e-06,
                                                               0.23102731],
                                                              [6.2e-07,
                                                               2.51e-06,
                                                               -0.48393604],
                                                              [2.43e-06,
                                                               1.68e-06,
                                                               0.48540866],
                                                              [8.2e-06,
                                                               -6.46e-06,
                                                               -0.23249993]]},
                                       'user': 'zulissi'},
             'intercept': 0.03917124250288029,
             'intercept_uncertainty': 0.010904477139158179,
             'miller': [1, 0, 0],
             'mongo_id': ObjectId('5ad4ca5b0ea601a4a743eb9a'),
             'mpid': 'mp-81'},
            {'FW_info': {'4': 236355, '6': 236360, '8': 236361, 'bulk': 218087},
             'formula': 'Na4',
             'initial_configuration': {'atoms': {'atoms': [{'charge': 0.0,
                                                            'index': 0,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.2358572150000002,
                                                                         1.74776604,
                                                                         22.70414684375],
                                                            'symbol': 'Na',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 1,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         19.676927385672116],
                                                            'symbol': 'Na',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 2,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [1.2358572150000002,
                                                                         1.74776604,
                                                                         16.649707564327883],
                                                            'symbol': 'Na',
                                                            'tag': 0},
                                                           {'charge': 0.0,
                                                            'index': 3,
                                                            'magmom': 0.0,
                                                            'momentum': [0.0, 0.0, 0.0],
                                                            'position': [0.0,
                                                                         0.0,
                                                                         13.62248810625],
                                                            'symbol': 'Na',
                                                            'tag': 0}],
                                                 'cell': [[3.70757165, -0.0, -0.0],
                                                          [-1.23585722, 3.49553208, 0.0],
                                                          [0.0, 0.0, 36.32663495]],
                                                 'chemical_symbols': ['Na'],
                                                 'constraints': [{'kwargs': {'indices': [1,
                                                                                         2]},
                                                                  'name': 'FixAtoms'}],
                                                 'info': {},
                                                 'mass': 91.95907712,
                                                 'natoms': 4,
                                                 'pbc': [True, True, True],
                                                 'spacegroup': 'Cmme (67)',
                                                 'symbol_counts': {'Na': 4},
                                                 'volume': 470.79085102330345},
                                       'calculator': {'class': 'SinglePointCalculator',
                                                      'module': 'ase.calculators.singlepoint'},
                                       'ctime': datetime.datetime(2018, 4, 16, 10, 47, 20, 949000),
                                       'inserted-hash': '4baf9441a3233242cb5f5cca17f46cbd35bba543',
                                       'mtime': datetime.datetime(2018, 4, 16, 10, 47, 20, 949000),
                                       'results': {'energy': -4.44411952,
                                                   'fmax': 0.09580518,
                                                   'fmax_unconstrained': 0.09580518,
                                                   'forces': [[1.39e-06,
                                                               1.35e-06,
                                                               -0.09554015],
                                                              [0.0, 0.0, 0.0],
                                                              [0.0, 0.0, 0.0],
                                                              [-4.1e-07,
                                                               -1.04e-06,
                                                               0.09580518]]},
                                       'user': 'zulissi'},
             'intercept': 0.016283841731280224,
             'intercept_uncertainty': 0.0008842929827786094,
             'miller': [1, 1, 0],
             'mongo_id': ObjectId('5ad4e94a28e39d37cc076c4b'),
             'mpid': 'mp-127'}]
    return docs


def test_hash_docs():
    '''
    Note that since Python 3's `hash` returns a different hash for
    each instance of Python, we actually perform regression testing
    ond the pre-hashed strings, not the hash itself.
    '''
    docs = get_expected_aggregated_adsorption_documents()

    # Ignore no keys
    strings = hash_docs(docs=docs, _return_hashes=False)
    expected_strings = [("adslab_calculation_date=2017-05-16 08:56:29.993000; adsorbates=['H']; "
                         'coordination=Al-Ni; energy=-0.17; formula=HAl6Ni10; miller=[0, 0, 1]; '
                         "mpid=mp-16514; neighborcoord=['Al:Ni-Ni-Ni-Ni-Ni-Ni-Ni', "
                         "'Ni:Al-Al-Al-Al-Al-Ni-Ni']; "
                         'nextnearestcoordination=Al-Al-Al-Al-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni; shift=0; '
                         'top=True; ')]
    assert strings == expected_strings

    # Ignore a key
    strings = hash_docs(docs=docs, ignore_keys=['mpid'], _return_hashes=False)
    expected_strings = ["adslab_calculation_date=2017-05-16 08:56:29.993000; adsorbates=['H']; "
                        'coordination=Al-Ni; energy=-0.17; formula=HAl6Ni10; miller=[0, 0, 1]; '
                        "neighborcoord=['Al:Ni-Ni-Ni-Ni-Ni-Ni-Ni', 'Ni:Al-Al-Al-Al-Al-Ni-Ni']; "
                        'nextnearestcoordination=Al-Al-Al-Al-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni; shift=0; '
                        'top=True; ']
    assert strings == expected_strings


def test_hash_doc():
    '''
    Note that since Python 3's `hash` returns a different hash for
    each instance of Python, we actually perform regression testing
    ond the pre-hashed string, not the hash itself.
    '''
    doc = get_expected_aggregated_adsorption_documents()[0]

    # Ignore no keys
    string = hash_doc(doc=doc, _return_hash=False)
    expected_string = ("adslab_calculation_date=2017-05-16 08:56:29.993000; adsorbates=['H']; "
                       'coordination=Al-Ni; energy=-0.17; formula=HAl6Ni10; miller=[0, 0, 1]; '
                       'mongo_id=59a015cbd3952577173b122d; mpid=mp-16514; '
                       "neighborcoord=['Al:Ni-Ni-Ni-Ni-Ni-Ni-Ni', 'Ni:Al-Al-Al-Al-Al-Ni-Ni']; "
                       'nextnearestcoordination=Al-Al-Al-Al-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni; shift=0; '
                       'top=True; ')
    assert string == expected_string

    # Ignore a key
    string = hash_doc(doc=doc, ignore_keys=['mpid'], _return_hash=False)
    expected_string = ("adslab_calculation_date=2017-05-16 08:56:29.993000; adsorbates=['H']; "
                       'coordination=Al-Ni; energy=-0.17; formula=HAl6Ni10; miller=[0, 0, 1]; '
                       "mongo_id=59a015cbd3952577173b122d; neighborcoord=['Al:Ni-Ni-Ni-Ni-Ni-Ni-Ni', "
                       "'Ni:Al-Al-Al-Al-Al-Ni-Ni']; "
                       'nextnearestcoordination=Al-Al-Al-Al-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni; shift=0; '
                       'top=True; ')
    assert string == expected_string
