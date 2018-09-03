'''
The function in this module is used to read the GASpy rc file. It is separated
from the rest of the module so that we don't need any non-native Python modules
to read the file.
'''

import os
from os.path import join
import json


def read_rc(query=None):
    '''
    This function will pull out keys from the .gaspyrc file for you

    Input:
        query   [Optional] The string indicating the configuration you want.
                If you're looking for nested information, use the syntax
                "foo.bar.key"
    Output:
        configs A dictionary whose keys are the input keys and whose values
                are the values that we found in the .gaspyrc file
    '''
    # Pull out the PYTHONPATH environment variable
    # so that we know where to look for the .gaspyrc file
    try:
        python_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        raise KeyError('You do not have the PYTHONPATH environment variable. You need to add GASpy to it')

    # Initializating our search for the .gaspyrc file
    rc_file = '.gaspyrc.json'
    rc_template = '.gaspyrc_template.json'
    found_config = False
    # Search our PYTHONPATH one-by-one
    for path in python_paths:
        for root, dirs, files in os.walk(path):
            if rc_file in files:
                rc_file = join(root, rc_file)
                found_config = True
                break
            # Warn the user if they haven't made a .gaspyrc.json file yet
            if rc_template in files:
                raise EnvironmentError('You have not yet made an appropriate .gaspyrc.json configuration file yet.')
        if found_config:
            break

    # Now that we've found it, open it up and read from it
    with open(rc_file, 'r') as rc:
        configs = json.load(rc)

    # Return out the keys you asked for. If the user did not specify the key, then return it all
    if query:
        keys = query.split('.')
        for key in keys:
            try:
                configs = configs[key]
            except KeyError as err:
                err.message += "; Check the spelling/capitalization of the config you're looking for"
                raise
        return configs
    else:
        return configs
