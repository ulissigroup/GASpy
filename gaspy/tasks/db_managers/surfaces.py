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

    # Create a `CalculateSurfaceEnergy` task for each surface energy
    # calculation that is in-progress.
    tasks = [CalculateSurfaceEnergy(mpid=mpid,
                                    miller_indices=miller_indices,
                                    shift=shift,
                                    vasp_settings={key: value for key, value in vasp_settings})
             for mpid, miller_indices, shift, vasp_settings in surfaces]

    # Run each task and then see which ones are done
    print('[%s] Calculating adsorption energies...' % datetime.now())
    multimap(__run_calculate_surface_energy_task, tasks,
             processes=n_processes, maxtasksperchild=10, chunksize=100,
             n_calcs=len(tasks))
    completed_tasks = [task for task in tasks if task.complete()]

    # Parse the completed tasks into documents for us to save
    print('[%s] Creating adsorption documents...' % datetime.now())
    surface_energy_docs = multimap(__create_surface_energy_doc,
                                   completed_tasks,
                                   processes=n_processes,
                                   maxtasksperchild=1,
                                   chunksize=100,
                                   n_calcs=len(completed_tasks))

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
        task    A list of `luigi.Task` objects, preferably ones from
                `gaspy.tasks.metadata_calculators.CalculateSurfaceEnergy`
    '''
    # Run each task again in case the relaxations are all done, but we just
    # haven't calculated the surface energy yet
    try:
        run_task(task)

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
                      % (task.mpid, task.miller_indices, task.shift,
                         unfreeze_dict(task.vasp_settings)))


def __create_surface_energy_doc(surface_energy_task):
    '''
    This function will create a Mongo document for the `surface_energy`
    collection given the output of the `CalculateSurfaceEnergy` task.

    Arg:
        surface_energy_task     An instance of a completed
                                `gaspy.tasks.metadata_calculators.CalculateSurfaceEnergy`
                                task
    Returns:
        doc     A modified form of the dictionary created by the
                `CalculateSurfaceEnergy` task. Will have the following keys:
                    surface_structures              Dictionaries for each of
                                                    the surfaces whose keys
                                                    indicate the number of
                                                    atoms and whose values are
                                                    the corresponding documents
                                                    found in the `atoms`
                                                    collection
                                                    `gaspy.mongo.make_doc_from_atoms`.
                    surface_energy                  A float indicating the
                                                    surface energy in
                                                    eV/Angstrom**2
                    surface_energy_standard_error   A float indicating the
                                                    standard error of our
                                                    estimate of the surface
                                                    energy
                    movement_data                   A dictionary whose keys are
                                                    the number of atoms of each
                                                    surface and whose values
                                                    are the maximum distance a
                                                    single atom moved during
                                                    relaxation.
                    fwids                           A dictionary whose keys are
                                                    the number of atoms of each
                                                    surface and whose values
                                                    are the FWID
                    calculation_dates               A dictionary whose keys are
                                                    the number of atoms of each
                                                    surface and whose values
                                                    are the date the calculation
                                                    finished.
                    fw_directories                  A dictionary whose keys are
                                                    the number of atoms of each
                                                    surface and whose values
                                                    are the directories where
                                                    the corresponding FireWork
                                                    ran
    '''
    # The output of the task to calculate surface energies will provide the
    # template for the document in our Mongo collection
    doc = get_task_output(surface_energy_task)
    doc['movement_data'] = {}
    doc['fwids'] = {}
    doc['calculation_dates'] = {}
    doc['fw_directories'] = {}

    # Figure out how far each of the structures moved during relaxation.
    for n_atoms, surface_doc in doc['surface_structures'].items():
        initial_atoms = make_atoms_from_doc(surface_doc['initial_configuration'])
        final_atoms = make_atoms_from_doc(surface_doc)
        max_movement = find_max_movement(initial_atoms, final_atoms)
        doc['movement_data'][n_atoms] = max_movement

        # Move some information from the individual surface documents to the
        # higher-level surface energy document
        doc['fwids'][n_atoms] = surface_doc['fwid']
        doc['calculation_dates'][n_atoms] = surface_doc['calculation_date']
        doc['fw_directories'][n_atoms] = surface_doc['directory']
        del surface_doc['fwid']
        del surface_doc['calculation_date']
        del surface_doc['directory']
        del surface_doc['fwname']   # This is just redundant information

    # Record the calculation settings
    doc['mpid'] = surface_energy_task.mpid
    doc['miller'] = surface_energy_task.miller_indices
    doc['shift'] = surface_energy_task.shift
    doc['vasp_settings'] = unfreeze_dict(surface_energy_task.vasp_settings)

    return doc
