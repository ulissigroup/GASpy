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
                     unfreeze_dict,
                     turn_site_into_str,
                     turn_string_site_into_tuple)

# Things we need to do the tests
import pytest
import collections
import json
from luigi.parameter import _FrozenOrderedDict


@pytest.mark.parametrize('query',
                         [None,
                          'gasdb_path',
                          'fireworks_info.lpad.host',
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


def test_turn_site_into_str():
    assert turn_site_into_str([0., 0., 0.]) == '[  0.     0.     0.  ]'
    assert turn_site_into_str([-0., 0., 0.]) == '[ -0.     0.     0.  ]'
    assert turn_site_into_str([1.23, 4.56, -7.89]) == '[  1.23   4.56  -7.89]'
    assert turn_site_into_str([10.23, -40.56, 70.89]) == '[ 10.23 -40.56  70.89]'
    assert turn_site_into_str([-10.23, -40.56, 70.89]) == '[-10.23 -40.56  70.89]'


def test_turn_string_site_into_tuple():
    assert turn_string_site_into_tuple('[  0.     0.     0.  ]') == (0., 0., 0.)
    assert turn_string_site_into_tuple('[ -0.     0.     0.  ]') == (-0., 0., 0.)
    assert turn_string_site_into_tuple('[  1.23   4.56  -7.89]') == (1.23, 4.56, -7.89)
    assert turn_string_site_into_tuple('[ 10.23 -40.56  70.89]') == (10.23, -40.56, 70.89)
    assert turn_string_site_into_tuple('[-10.23 -40.56  70.89]') == (-10.23, -40.56, 70.89)
