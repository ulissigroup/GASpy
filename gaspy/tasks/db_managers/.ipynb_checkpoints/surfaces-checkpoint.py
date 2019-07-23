'''
This module contains functions we use to update our `surface_energy` Mongo
collection, which contains surface energy calculations and associated
information.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import traceback
import warnings
from datetime import datetime
import luigi
from ..core import get_task_output, run_task
from ..calculation_finders import FindBulk
from ..metadata_calculators import CalculateSurfaceEnergy
from ...utils import unfreeze_dict, multimap
from ...gasdb import get_mongo_collection
from ...mongo import make_atoms_from_doc
from ...atoms_operators import find_max_movement

def update_surface_energy_collection(n_processes=1):
    '''
    This function will parse and dump all of the completed surface energy
    calculations in our `atoms` Mongo collection into our `surface_energy`
    collection. It will not dump anything that is already there.

    Args:
        n_processes     An integer indicating how many threads you want to use
                        when running the tasks. If you do not expect many
                        updates, stick to the default of 1, or go up to 4. If
                        you are re-creating your collection from scratch, you
                        may want to want to increase this argument as high as
                        you can.
    '''
    # Identify the surfaces that have been at least partially calculated, but
    # not yet added to the surface energy collection
    atoms_docs = _find_atoms_docs_not_in_surface_energy_collection()
    surfaces = set()
    for doc in atoms_docs:
        mpid = doc['fwname']['mpid']
        miller_indices = tuple(doc['fwname']['miller'])
        shift = round(doc['fwname']['shift'], 3)
        # We'll need to make the vasp_settings hashable
        vasp_settings = doc['fwname']['vasp_settings']
        vasp_settings['kpts'] = tuple(vasp_settings['kpts'])  # make hashable
        vasp_settings = tuple((key, value) for key, value in vasp_settings.items())
        # Define a surface according to mpid, miller, shift, and calculation
        # settings
        surface = (mpid, miller_indices, shift, vasp_settings)
        surfaces.add(surface)

    # Create a list of tasks that has `CalculateSurfaceEnergy` task 
    # calculation that is in-progress.
    surface_tasks = [CalculateSurfaceEnergy(mpid=mpid,
                                    miller_indices=miller_indices,
                                    shift=shift,
                                    vasp_settings={key: value for key, value in vasp_settings})
             for mpid, miller_indices, shift, vasp_settings in surfaces]
    # For each 'CalculateSurfaceEnergy', also have a FindBulk task 
    # and make them into a list 
    tasks = [[task, FindBulk(mpid=mpid, vasp_settings = task.bulk_vasp_settings)] for task in surface_tasks]

    # Run each task and then see which ones' slab relaxation are done
    print('[%s] Calculating surface energies...' % datetime.now())
    multimap(__run_calculate_surface_energy_task, tasks,
             processes=n_processes, maxtasksperchild=10, chunksize=100,
             n_calcs=len(tasks))
    completed_tasks = [task for task in tasks if task[0].complete()]

    # Parse the completed tasks into documents for us to save
    print('[%s] Creating surface energy documents...' % datetime.now())
    surface_energy_docs = multimap(__create_surface_energy_doc,
                                   completed_tasks,
                                   processes=n_processes,
                                   maxtasksperchild=1,
                                   chunksize=100,
                                   n_calcs=len(completed_tasks))

    if len(surface_energy_docs) > 0:
        print('[%s] Creating %i new entries in the surface energy collection...'
              % (datetime.now(), len(surface_energy_docs)))
        with get_mongo_collection('surface_energy') as collection:
            collection.insert_many(surface_energy_docs)
        print('[%s] Created %i new entries in the surface energy collection'
              % (datetime.now(), len(surface_energy_docs)))


def _find_atoms_docs_not_in_surface_energy_collection():
    '''
    This function will get the Mongo documents of surface energy calculations
    that are inside our `atoms` collection, but not inside our `surface_energy`
    collection.

    Returns:
        missing_docs    A list of surface energy documents from the `atoms`
                        collection that have not yet been added to the
                        `surface_energy` collection.
    '''
    # Find the FWIDs of the documents inside our surface energy collection
    with get_mongo_collection('surface_energy') as collection:
        surface_energy_docs = list(collection.find({}, {'fwids': 'fwids', '_id': 0}))
    fwids_in_se = {fwid for doc in surface_energy_docs for fwid in doc['fwids']}

    # Find the FWIDs of the documents inside our atoms collection
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'surface energy optimization'}
        projection = {'fwid': 'fwid', '_id': 0}
        docs_atoms = list(collection.find(query, projection))
        fwids_in_atoms = {doc['fwid'] for doc in docs_atoms}

        # Pull the atoms documents of everything that's missing from our
        # adsorption collection. Although we use `find` a second time, this
        # time we are getting the whole document (not just the FWID), so we
        # only want to do this for the things we need.
        fwids_missing = fwids_in_atoms - fwids_in_se
        missing_docs = list(collection.find({'fwid': {'$in': list(fwids_missing)}}))
    return missing_docs


def __run_calculate_surface_energy_task(task):
    '''
    This function will run some tasks for you and return the ones that
    successfully completed.

    Args:
        task    A list of lists that contains `luigi.Task` objects, 
                preferably ones from
                `gaspy.tasks.metadata_calculators.CalculateSurfaceEnergy` and
                `gaspy.tasks.calculation_finders.FindBulk`
    '''
    # Run each task again in case the relaxations are all done, but we just
    # haven't calculated the surface energy yet
    slab_task = task[0]
    bulk_task = task[1]
    
    try:
        run_task(slab_task) 
        run_task(bulk_task) 

    # If a task has failed and not produced an output, we don't want that to
    # stop us from updating the successful runs.
    except FileNotFoundError:
        pass

    # If the output already exists, then move on
    except luigi.target.FileAlreadyExists:
        pass

    # If some other error pops up, then we want to report it. But we also want
    # to move on so that we can still update other things.
    except:     # noqa: E722
        traceback.print_exc()
        warnings.warn('We caught the exception reported just above and '
                      'moved on without updating the collection. Here is '
                      'the offending surface energy calculation information: '
                      ' (%s, %s, %s, %s)'
                      % (slab_task.mpid, slab_task.miller_indices, slab_task.shift,
                         unfreeze_dict(slab_task.vasp_settings)))


def __create_surface_energy_doc(surface_energy_task_list):
    '''
    This function will create a Mongo document for the `surface_energy`
    collection given the output of the `CalculateSurfaceEnergy` task.

    Arg:
        surface_energy_tasks    A list that contains an instance of a completed
                                `gaspy.tasks.metadata_calculators.CalculateSurfaceEnergy`
                                task and its bulk relxation task
    Returns:
        doc     A modified form of the dictionary created by the
                `CalculateSurfaceEnergy` task. Will have the following keys:
                    surface_structures              A list of three
                                                    dictionaries for each of
                                                    the surfaces. These
                                                    dictionaries are the
                                                    documents found in the
                                                    `atoms` collection
                                                    `gaspy.mongo.make_doc_from_atoms`.
                    surface_energy                  A float indicating the
                                                    surface energy in
                                                    eV/Angstrom**2
                    surface_energy_standard_error   A float indicating the
                                                    standard error of our
                                                    estimate of the surface
                                                    energy
                    max_atom_movement               A list containing three
                                                    floats, where each float is
                                                    the maximum distance a
                                                    single atom moved during
                                                    relaxation for each of the
                                                    three surfaces.
                    fwids                           A dictionary containing 2 items.
                                                    One is a list contains three
                                                    integers, where each
                                                    integer is the FireWork ID
                                                    of each of the three
                                                    surfaces. Another is the fwid of 
                                                    the bulk relxation.
                    calculation_dates               A list containing three
                                                    `datetime.datetime`
                                                    objects, where each
                                                    datetime is the date that
                                                    each of the three surface
                                                    calculations were
                                                    completed.
                    fw_directories                  A dictionary containing 2 items.
                                                    One is a list containing three
                                                    strings corresponding to
                                                    the FireWorks directories
                                                    where each of the surface
                                                    relaxations were performed.
                                                    Another is the string of bulk relxation 
                                                    fw directory. 
    '''
    # The output of the task to calculate surface energies will provide the
    # template for the document in our Mongo collection
    surface_energy_task = surface_energy_task_list[0]
    bulk_relaxation_task = surface_energy_task_list[1]
    doc = get_task_output(surface_energy_task)
    doc['max_atom_movement'] = []
    doc['fwids'] = {'slabs':[], 'bulk':''}
    doc['calculation_dates'] = []
    doc['fw_directories'] = {'slabs':[], 'bulk':''}

    # Figure out how far each of the structures moved during relaxation.
    for surface_doc in doc['surface_structures']:
        initial_atoms = make_atoms_from_doc(surface_doc['initial_configuration'])
        final_atoms = make_atoms_from_doc(surface_doc)
        max_movement = find_max_movement(initial_atoms, final_atoms)
        doc['max_atom_movement'].append(max_movement)

        # Move some information from the individual surface documents to the
        # higher-level surface energy document
        doc['fwids']['slabs'].append(surface_doc['fwid'])
        doc['calculation_dates'].append(surface_doc['calculation_date'])
        doc['fw_directories']['slabs'].append(surface_doc['directory'])
        del surface_doc['fwid']
        del surface_doc['calculation_date']
        del surface_doc['directory']
        del surface_doc['fwname']   # This is just redundant information

    # Record the calculation settings
    doc['mpid'] = surface_energy_task.mpid
    doc['miller'] = surface_energy_task.miller_indices
    doc['shift'] = surface_energy_task.shift
    doc['vasp_settings'] = unfreeze_dict(surface_energy_task.vasp_settings)
    
    # Add the bulk relaxation fwid and fw_directory
    bulk_doc = get_task_output(bulk_relaxation_task)
    doc['fwids']['bulk'] = bulk_doc['fwid']
    doc['fw_directories']['bulk'] = bulk_doc['directory']
    return doc
