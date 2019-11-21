''' Various functions that may be used across GASpy and its submodules '''

__authors__ = ['Kevin Tran', 'Zack Ulissi']
__emails__ = ['ktran@andrew.cmu.edu', 'zulissi@andrew.cmu.edu']

import gc
import os
import json
import numpy as np
from multiprocess import Pool
from collections import OrderedDict, Iterable, Mapping
from tqdm import tqdm


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


def multimap(function, inputs, chunked=False, processes=32, maxtasksperchild=1,
             chunksize=1, n_calcs=None, desc=None):
    '''
    This function is a wrapper to parallelize a function.

    Args:
        function            The function you want to execute
        inputs              An iterable that yields proper arguments to the
                            function
        chunked             A Boolean indicating whether your function expects
                            single arguments or "chunked" iterables, e.g.,
                            lists.
        processes           The number of threads/processes you want to be using
        maxtasksperchild    The maximum number of tasks that a child process
                            may do before terminating (and therefore clearing
                            its memory cache to avoid memory overload).
        chunksize           How many calculations you want to have each single
                            processor do per task. Smaller chunks means more
                            memory shuffling. Bigger chunks means more RAM
                            requirements.
        n_calcs             How many calculations you have. Only necessary for
                            adding a percentage timer to the progress bar.
        desc                String indicating what you want the TQDM label to
                            be for the progress bar
    Returns:
        outputs     A list of the inputs mapped through the function
    '''
    # Collect garbage before we begin multiprocessing to make sure we don't
    # pass things we don't need to
    gc.collect()

    # If we have one thread, there's no use multiprocessing
    if processes == 1:
        output = [function(input_) for input_ in tqdm(inputs, total=n_calcs, desc=desc)]
        return output

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        # Use multiprocessing to perform the calculations. We use imap instead
        # of map so that we get an iterator, which we need for tqdm (the
        # progress bar) to work. imap also requires less disk memory, which
        # can be an issue for some of our large systems.
        if not chunked:
            iterator = pool.imap(function, inputs, chunksize=chunksize)
            total = n_calcs
            outputs = list(tqdm(iterator, total=total, desc=desc))

        # If our function expects chunks, then we have to unpack our inputs
        # appropriately
        else:
            iterator = pool.imap(function, _chunk(inputs, n=chunksize))
            total = n_calcs / chunksize
            outputs = list(np.concatenate(list(tqdm(iterator, total=total, desc=desc))))

    return outputs


def _chunk(iterable, n):
    '''
    Takes an iterable and then gives you a generator that yields chunked lists
    of the iterable.

    Args:
        iterable    Any iterable object
        n           An integer indicating the size of the lists you want
                    returned
    Returns:
        generator   Python generator that yields lists of size `n` with the
                    same contents as the `iterable` you passed in.
    '''
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def multimap_method(instance, method, inputs, chunked=False, processes=32,
                    maxtasksperchild=1, chunksize=1, n_calcs=None, **kwargs):
    '''
    This function pools and maps methods of class instances. It does so by
    putting the class instance into the global space so that each worker can
    pull it. This prevents the multiprocessor from pickling/depickling the
    instance for each worker, thus saving time. This is especially needed for
    lazy learning models like GP.

    Args:
        instance            An instance of a class
        method              A string indicating the method of the class that
                            you want to map
        function            The function you want to execute
        inputs              An iterable that yields proper arguments to the
                            function
        chunked             A Boolean indicating whether your function expects
                            single arguments or "chunked" iterables, e.g.,
                            lists.
        processes           The number of threads/processes you want to be using
        maxtasksperchild    The maximum number of tasks that a child process
                            may do before terminating (and therefore clearing
                            its memory cache to avoid memory overload).
        chunksize           How many calculations you want to have each single
                            processor do per task. Smaller chunks means more
                            memory shuffling. Bigger chunks means more RAM
                            requirements.
        n_calcs             How many calculations you have. Only necessary for
                            adding a percentage timer to the progress bar.
        kwargs              Any arguments that you should be passing to the
                            method
    Returns:
        outputs     A list of the inputs mapped through the function
    '''
    # Push the class instance to global space for process sharing.
    global module_instance
    module_instance = instance

    # Full the method out of the class instance
    def function(arg):  # noqa: E306
        return getattr(module_instance, method)(arg, **kwargs)

    # Call the `multimap` function for the rest of the work
    outputs = multimap(function, inputs, chunked=chunked,
                       processes=processes,
                       maxtasksperchild=maxtasksperchild,
                       chunksize=chunksize,
                       n_calcs=n_calcs)

    # Clean up and output
    del globals()['module_instance']
    return outputs
