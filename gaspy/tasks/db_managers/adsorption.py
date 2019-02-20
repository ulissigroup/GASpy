'''
This module houses various functions to update our `adsorption` Mongo
collection, which contains adsorption energies and various associated
information.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import traceback
import warnings
import multiprocess
from ..core import get_task_output, evaluate_luigi_task
from ..metadata_calculators import CalculateAdsorptionEnergy
from ...mongo import make_atoms_from_doc, make_doc_from_atoms
from ...gasdb import get_mongo_collection
from ...atoms_operators import fingerprint_adslab, find_max_movement


def update_adsorption_collection(n_processes=1):
    '''
    This function will parse and dump all of the completed adsorption
    calculations in our `atoms` Mongo collection into our `adsorption`
    collection. It will not dump anything that is already there.

    Args:
        n_processes     An integer indicating how many threads you want to use
                        when running the tasks. If you do not expect many
                        updates, stick to the default of 1, or go up to 4. If
                        you are re-creating your collection from scratch, you
                        may want to want to increase this argument as high as
                        you can.
    '''
    # Figure out what we need to dump
    missing_docs = _find_atoms_docs_not_in_adsorption_collection()

    # Multi-thread calculations for adsorption energies
    if n_processes > 1:
        with multiprocess.Pool(n_processes) as pool:
            generator = pool.imap(__run_calculate_adsorption_energy_task,
                                  missing_docs, chunksize=100)
            calc_energy_docs = list(generator)
    # Don't need pooling if we only want to use one thread
    else:
        calc_energy_docs = [__run_calculate_adsorption_energy_task(doc)
                            for doc in missing_docs]

    # If a calculation fails, our helper function will return `None`. Let's
    # take those out here.
    calc_energy_docs = [doc for doc in calc_energy_docs if doc is not None]

    # Turn the adsorption energies into `adsorption` documents, then save them
    adsorption_docs = [__create_adsorption_doc(doc) for doc in calc_energy_docs]
    if len(adsorption_docs) > 0:
        with get_mongo_collection('adsorption') as collection:
            collection.insert_many(adsorption_docs)
        print('Just created %i new entries in the adsorption collection'
              % len(adsorption_docs))


def _find_atoms_docs_not_in_adsorption_collection():
    '''
    This function will get the Mongo documents of adsorption calculations that
    are inside our `atoms` collection, but not inside our `adsorption`
    collection.

    Returns:
        missing_ads_docs    A list of adsorption documents from the `atoms`
                            collection that have not yet been added to the
                            `adsorption` collection.
    '''
    # Find the FWIDs of the documents inside our adsorption collection
    with get_mongo_collection('adsorption') as collection:
        docs_adsorption = list(collection.find({}, {'fwids': 'fwids', '_id': 0}))
    fwids_in_adsorption = set(doc['fwids']['slab+adsorbate'] for doc in docs_adsorption)

    # Find the FWIDs of the documents inside our atoms collection
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'slab+adsorbate optimization',
                 'fwname.adsorbate': {'$ne': ''}}
        projection = {'fwid': 'fwid', '_id': 0}
        docs_atoms = list(collection.find(query, projection))
        fwids_in_atoms = set(doc['fwid'] for doc in docs_atoms)

        # Pull the atoms documents of everything that's missing from our
        # adsorption collection. Although we use `find` a second time, this
        # time we are getting the whole document (not just the FWID), so we
        # only want to do this for the things we need.
        fwids_missing = fwids_in_atoms - fwids_in_adsorption
        missing_ads_docs = list(collection.find({'fwid': {'$in': list(fwids_missing)}}))
    return missing_ads_docs


def __run_calculate_adsorption_energy_task(atoms_doc):
    '''
    This function will parse adsorption documents from our `atoms` collection,
    create Luigi tasks to calculate adsorption energies for each document, run
    the tasks, and then give you the results.

    Args:
        atoms_doc   A dictionary taken from our `atoms` Mongo collection
    Returns:
        energy_doc  A dictionary obtained from the output of the
                    `CalculateAdsorptionEnergy` task
    '''
    # Reformat the site because of silly historical reasons
    adsorption_site = atoms_doc['fwname']['adsorption_site']

    # Create, run, and return the output of the task
    task = CalculateAdsorptionEnergy(adsorption_site=adsorption_site,
                                     shift=atoms_doc['fwname']['shift'],
                                     top=atoms_doc['fwname']['top'],
                                     adsorbate_name=atoms_doc['fwname']['adsorbate'],
                                     rotation=atoms_doc['fwname']['adsorbate_rotation'],
                                     mpid=atoms_doc['fwname']['mpid'],
                                     miller_indices=atoms_doc['fwname']['miller'],
                                     adslab_vasp_settings=atoms_doc['fwname']['vasp_settings'])
    try:
        evaluate_luigi_task(task)
        energy_doc = get_task_output(task)
        return energy_doc

    # If a task has failed and not produced an output, we don't want that to
    # stop us from updating the successful runs
    except FileNotFoundError:
        pass

    # If some other error pops up, then we want to report it, but move on so
    # that we can still update other things.
    except RuntimeError:
        traceback.print_exc()
        warnings.warn('We caught the exception reported just above and moved on '
                      'with updating the adsorption updating. Here is the '
                      'offending document:\n%s' % atoms_doc)


def __create_adsorption_doc(energy_doc):
    '''
    This function will create a Mongo document for the `adsorption` collection
    given the output of the `CalculateAdsorptionEnergy` task.

    Arg:
        energy_doc  A dictionary created by the `CalculateAdsorptionEnergy`
                    task
    '''
    # Get the `atoms` documents for the slab and adslab
    with get_mongo_collection('atoms') as collection:
        adslab_doc = list(collection.find({'fwid': energy_doc['fwids']['adslab']}))[0]
        slab_doc = list(collection.find({'fwid': energy_doc['fwids']['slab']}))[0]

    # Get some pertinent `ase.Atoms` objects
    bare_slab_init = make_atoms_from_doc(slab_doc['initial_configuration'])
    bare_slab_final = make_atoms_from_doc(slab_doc)
    adslab_init = make_atoms_from_doc(adslab_doc['initial_configuration'])
    adslab_final = make_atoms_from_doc(adslab_doc)
    # In GASpy, atoms tagged with 0's are slab atoms. Atoms tagged with
    # integers > 0 are adsorbates. We use that information to pull our the slab
    # and adsorbate portions of the adslab.
    adsorbate_init = adslab_init[adslab_init.get_tags() > 0]
    adsorbate_final = adslab_final[adslab_final.get_tags() > 0]
    slab_init = adslab_init[adslab_init.get_tags() == 0]
    slab_final = adslab_final[adslab_final.get_tags() == 0]

    # Fingerprint the adslab before and after relaxation
    fp_init = fingerprint_adslab(adslab_init)
    fp_final = fingerprint_adslab(adslab_final)

    # Figure out how far the bare slab moved during relaxation. Then do the
    # same for the slab and the adsorbate.
    max_bare_slab_movement = find_max_movement(bare_slab_init, bare_slab_final)
    max_slab_movement = find_max_movement(slab_init, slab_final)
    max_ads_movement = find_max_movement(adsorbate_init, adsorbate_final)

    # Parse the data into a Mongo document
    adsorption_doc = make_doc_from_atoms(adslab_final)
    adsorption_doc['initial_configuration'] = make_doc_from_atoms(adslab_init)
    adsorption_doc['adsorption_energy'] = energy_doc['adsorption_energy']
    adsorption_doc['adsorbate'] = adslab_doc['fwname']['adsorbate']
    adsorption_doc['adsorbate_rotation'] = adslab_doc['fwname']['adsorbate_rotation']
    adsorption_doc['initial_adsorption_site'] = adslab_doc['fwname']['adsorption_site']
    adsorption_doc['mpid'] = adslab_doc['fwname']['mpid']
    adsorption_doc['miller'] = adslab_doc['fwname']['miller']
    adsorption_doc['shift'] = adslab_doc['fwname']['shift']
    adsorption_doc['top'] = adslab_doc['fwname']['top']
    adsorption_doc['slab_repeat'] = adslab_doc['fwname']['slab_repeat']
    adsorption_doc['vasp_settings'] = adslab_doc['fwname']['vasp_settings']
    adsorption_doc['fwids'] = {'slab+adsorbate': adslab_doc['fwid'],
                               'slab': slab_doc['fwid']}
    adsorption_doc['fp_final'] = fp_final
    adsorption_doc['fp_init'] = fp_init
    adsorption_doc['movement_data'] = {'max_bare_slab_movement': max_bare_slab_movement,
                                       'max_slab_movement': max_slab_movement,
                                       'max_adsorbate_movement': max_ads_movement}
    return adsorption_doc
