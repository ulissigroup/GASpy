'''
This module houses various functions to update our `adsorption` Mongo
collection, which contains adsorption energies and various associated
information.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

from ..core import run_tasks, get_task_output
from ..metadata_calculators import CalculateAdsorptionEnergy
from ...utils import turn_string_site_into_tuple
from ...mongo import make_atoms_from_doc, make_doc_from_atoms
from ...gasdb import get_mongo_collection
from ...atoms_operators import fingerprint_adslab, find_max_movement


def update_adsorption_collection(workers=1, local_scheduler=False):
    '''
    This function will parse and dump all of the completed adsorption
    calculations in our `atoms` Mongo collection into our `adsorption`
    collection. It will not dump anything that is already there.

    Args:
        workers         An integer indicating how many processes/workers you
                        want executing the prerequisite Luigi tasks.
        local_scheduler A Boolean indicating whether or not you want to
                        use a local scheduler. You should use a local
                        scheduler only when you want something done
                        quickly but dirtily. If you do not use local
                        scheduling, then we will use our Luigi daemon
                        to manage things, which should be the status
                        quo.
    '''
    # Calculate the adsorption energies for adsorption calculations that show
    # up in our `atoms` Mongo collection
    missing_docs = _find_atoms_docs_not_in_adsorption_collection()
    calc_energy_docs = __get_luigi_adsorption_energies(missing_docs, workers, local_scheduler)

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
    fwids_in_adsorption = set(doc['fwids']['adsorption'] for doc in docs_adsorption)

    # Find the FWIDs of the documents inside our atoms collection
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'slab+adsorbate optimization'}
        projection = {'fwid': 'fwid', '_id': 0}
        docs_atoms = list(collection.find(query, projection))
        fwids_in_atoms = set(doc['fwid'] for doc in docs_atoms)

        # Pull the atoms documents of everything that's missing from our
        # adsorption collection. Although we use `find` a second time, this
        # time we are getting the whole document (not just the FWID), so we
        # only want to do this for the things we need.
        fwids_missing = fwids_in_atoms - fwids_in_adsorption
        missing_ads_docs = list(collection.find({'fw_id': {'$in': list(fwids_missing)}}))
    return missing_ads_docs


def __get_luigi_adsorption_energies(atoms_docs, workers=1, local_scheduler=False):
    '''
    This function will parse adsorption documents from our `atoms` collection,
    create Luigi tasks to calculate adsorption energies for each document, run
    the tasks, and then give you the results.

    Args:
        atoms_docs      A list of dictionaries taken from our `atoms` Mongo
                        collection
        workers         An integer indicating how many processes/workers you
                        want executing the prerequisite Luigi tasks.
        local_scheduler A Boolean indicating whether or not you want to
                        use a local scheduler. You should use a local
                        scheduler only when you want something done
                        quickly but dirtily. If you do not use local
                        scheduling, then we will use our Luigi daemon
                        to manage things, which should be the status
                        quo.
    Returns:
        energy_docs     A list of dictionaries obtained from the output of the
                        `CalculateAdsorptionEnergy` task
    '''
    # Make and execute the Luigi tasks to calculate the adsorption energies
    calc_energy_tasks = []
    for doc in atoms_docs:
        adsorption_site = turn_string_site_into_tuple(doc['fwname']['adsorption_site'])
        task = CalculateAdsorptionEnergy(adsorption_site=adsorption_site,
                                         shift=doc['fwname']['shift'],
                                         top=doc['fwname']['top'],
                                         adsorbate_name=doc['fwname']['adsorbate'],
                                         rotation=doc['fwname']['adsorbate_rotation'],
                                         mpid=doc['fwname']['mpid'],
                                         miller_indices=doc['fwname']['miller'],
                                         adslab_vasp_settings=doc['fwname']['vasp_settings'])
        calc_energy_tasks.append(task)
    run_tasks(calc_energy_tasks, workers=workers, local_scheduler=local_scheduler)

    # Get all of the results from each task
    energy_docs = []
    for task in calc_energy_tasks:
        try:
            energy_docs.append(get_task_output(task))

        # If a task has failed and not produced an output, we don't want that
        # to stop us from updating the successful runs
        except FileNotFoundError:
            continue

    return energy_docs


def __create_adsorption_doc(energy_doc):
    '''
    This function will create a Mongo document for the `adsorption` collection
    given the output of the `CalculateAdsorptionEnergy` task.

    Arg:
        energy_doc  A dictionary created by the `CalculateAdsorptionEnergy`
                    task
    '''
    # Get some pertinent `ase.Atoms` objects
    bare_slab_init = make_atoms_from_doc(energy_doc['slab']['initial_configuration'])
    bare_slab_final = make_atoms_from_doc(energy_doc['slab'])
    adslab_init = make_atoms_from_doc(energy_doc['adslab']['initial_configuration'])
    adslab_final = make_atoms_from_doc(energy_doc['adslab'])
    # In GASpy, atoms tagged with 0's are slab atoms. Atoms tagged with
    # integers > 0 are adsorbates. We use that information to pull our the slab
    # and adsorbate portions of the adslab.
    adsorbate_init = adslab_init[adslab_init.get_tags > 0]
    adsorbate_final = adslab_final[adslab_final.get_tags > 0]
    slab_init = adslab_init[adslab_init.get_tags == 0]
    slab_final = adslab_final[adslab_final.get_tags == 0]

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
    adsorption_doc['adsorbate'] = energy_doc['adslab']['fwname']['adsorbate']
    adsorption_doc['adsorbate_rotation'] = energy_doc['adslab']['fwname']['adsorbate_rotation']
    adsorption_doc['mpid'] = energy_doc['adslab']['fwname']['mpid']
    adsorption_doc['miller'] = energy_doc['adslab']['fwname']['miller']
    adsorption_doc['shift'] = energy_doc['adslab']['fwname']['shift']
    adsorption_doc['top'] = energy_doc['adslab']['fwname']['top']
    adsorption_doc['slabrepeat'] = energy_doc['adslab']['fwname']['slabrepeat']
    adsorption_doc['vasp_settings'] = energy_doc['adslab']['fwname']['vasp_settings']
    adsorption_doc['fwids'] = {'slab+adsorbate': energy_doc['adslab']['fwid'],
                               'slab': energy_doc['slab']['fwid']}
    adsorption_doc['fp_final'] = fp_final
    adsorption_doc['fp_init'] = fp_init
    adsorption_doc['movement_data'] = {'max_bare_slab_movement': max_bare_slab_movement,
                                       'max_slab_movement': max_slab_movement,
                                       'max_adsorbate_movement': max_ads_movement}
    return adsorption_doc
