''' Various functions that may be used across GASpy and its submodules '''

__authors__ = ['Kevin Tran', 'Zack Ulissi']
__emails__ = ['ktran@andrew.cmu.edu', 'zulissi@andrew.cmu.edu']

import os
import json
from collections import OrderedDict, Iterable, Mapping


def print_dict(dict_, indent=0):
    '''
    This function prings a nested dictionary, but in a prettier format. This is
    strictly for reporting and/or debugging purposes.

    Inputs:
        dict_   The nested dictionary to print
        indent  How many tabs to start the printing at
    '''
    if isinstance(dict_, dict):
        for key, value in dict_.items():
            # If the dictionary key is `spec`, then it's going to print out a
            # bunch of messy looking things we don't care about. So skip it.
            if key != 'spec':
                print('\t' * indent + str(key))
                if isinstance(value, dict) or isinstance(value, list):
                    print_dict(value, indent+1)
                else:
                    print('\t' * (indent+1) + str(value))
    elif isinstance(dict_, list):
        for item in dict_:
            if isinstance(item, dict) or isinstance(item, list):
                print_dict(item, indent+1)
            else:
                print('\t' * (indent+1) + str(item))
    else:
        pass


def read_rc(query=None):
    '''
    This function will pull out keys from the .gaspyrc file for you

    Input:
        query   [Optional] The string indicating the configuration you want.
                If you're looking for nested information, use the syntax
                "foo.bar.key"
    Output:
        rc_contents  A dictionary whose keys are the input keys and whose values
                     are the values that we found in the .gaspyrc file
    '''
    rc_file = _find_rc_file()
    with open(rc_file, 'r') as file_handle:
        rc_contents = json.load(file_handle)

    # Return out the keys you asked for. If the user did not specify the key, then return it all
    if query:
        keys = query.split('.')
        for key in keys:
            try:
                rc_contents = rc_contents[key]
            except KeyError as error:
                raise KeyError('Check the spelling/capitalization of the key/values you are looking for').with_traceback(error.__traceback__)

    return rc_contents


def _find_rc_file():
    '''
    This function will search your PYTHONPATH and look for the location of
    your .gaspyrc.json file.

    Returns:
        rc_file     A string indicating the full path to the first .gaspyrc.json
                    file it finds in your PYTHONPATH.
    '''
    # Pull out the PYTHONPATH environment variable
    # so that we know where to look for the .gaspyrc file
    try:
        python_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        raise EnvironmentError('You do not have the PYTHONPATH environment variable. You need to add GASpy to it')

    # Search our PYTHONPATH one-by-one
    rc_file = '.gaspyrc.json'
    for path in python_paths:
        for root, dirs, files in os.walk(path):
            if rc_file in files:
                rc_file = os.path.join(root, rc_file)
                return rc_file

            # If we can find the .gaspyrc_template.json file but not
            # a .gaspyrc.json file yet, then the user probably
            # has their PYTHONPATH set up correctly, but not their
            # .gaspyrc.json file set up, yet.
            if '.gaspyrc_template.json' in files:
                raise EnvironmentError('You have not yet made an appropriate .gaspyrc.json configuration file yet.')


def unfreeze_dict(frozen_dict):
    '''
    Recursive function to turn a Luigi frozen dictionary into an ordered dictionary,
    along with all of the branches.

    Arg:
        frozen_dict     Instance of a luigi.parameter._FrozenOrderedDict
    Returns:
        dict_   Ordered dictionary
    '''
    # If the argument is a dictionary, then unfreeze it
    if isinstance(frozen_dict, Mapping):
        unfrozen_dict = OrderedDict(frozen_dict)

        # Recur
        for key, value in unfrozen_dict.items():
            unfrozen_dict[key] = unfreeze_dict(value)

    # Recur on the object if it's a tuple
    elif isinstance(frozen_dict, tuple):
        unfrozen_dict = tuple(unfreeze_dict(element) for element in frozen_dict)

    # Recur on the object if it's a mutable iterable
    elif isinstance(frozen_dict, Iterable) and not isinstance(frozen_dict, str):
        unfrozen_dict = frozen_dict
        for i, element in enumerate(unfrozen_dict):
            unfrozen_dict[i] = unfreeze_dict(element)

    # If the argument is neither mappable nor iterable, we'rpe probably at a leaf
    else:
        unfrozen_dict = frozen_dict

    return unfrozen_dict
