'''
This module houses various functions to update our `adsorption` Mongo
collection, which contains adsorption energies and various associated
information.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import traceback
import warnings
import pprint
from datetime import datetime
import luigi
from ..core import get_task_output, run_task
from ..metadata_calculators import CalculateAdsorptionEnergy, CalculateRismAdsorptionEnergy
from ...defaults import DFT_CALCULATOR, gas_settings
from ...utils import multimap
from ...mongo import make_atoms_from_doc, make_doc_from_atoms
from ...gasdb import get_mongo_collection
from ...atoms_operators import fingerprint_adslab, find_max_movement


def update_adsorption_collection(dft_calculator=DFT_CALCULATOR, n_processes=1):
    '''
    This function will parse and dump all of the completed adsorption
    calculations in our `atoms` Mongo collection into our `adsorption`
    collection. It will not dump anything that is already there.

    Args:
        dft_calculator  A string indicating which DFT calculator you want to
                        parse data for---e.g., 'vasp', 'qe', or 'rism'. This
                        function will find the appropriate data and update
                        the appropriate adsorption collection.
        n_processes     An integer indicating how many threads you want to use
                        when running the tasks. If you do not expect many
                        updates, stick to the default of 1, or go up to 4. If
                        you are re-creating your collection from scratch, you
                        may want to want to increase this argument as high as
                        you can.
    '''
    # Figure out what we need to dump
    missing_docs = _find_atoms_docs_not_in_adsorption_collection(dft_calculator)

    # Calculate adsorption energies
    print('[%s] Calculating adsorption_%s energies...'
          % (datetime.now(), dft_calculator))
    calc_energy_docs = multimap(__run_calculate_adsorption_energy_task,
                                missing_docs, processes=n_processes,
                                maxtasksperchild=10, chunksize=100,
                                n_calcs=len(missing_docs))
    # Clean up
    cleaned_calc_energy_docs = __clean_calc_energy_docs(calc_energy_docs,
                                                        missing_docs)

    # Turn the adsorption energies into `adsorption` documents, then save them
    print('[%s] Creating adsorption_%s documents...'
          % (datetime.now(), dft_calculator))
    adsorption_docs = multimap(__create_adsorption_doc,
                               cleaned_calc_energy_docs,
                               processes=n_processes,
                               maxtasksperchild=1,
                               chunksize=100,
                               n_calcs=len(cleaned_calc_energy_docs))

    # Now write the documents
    if len(adsorption_docs) > 0:
        print('[%s] Creating %i new entries in the adsorption_%s collection...'
              % (datetime.now(), len(adsorption_docs), dft_calculator))
        with get_mongo_collection('adsorption_%s' % dft_calculator) as collection:
            collection.insert_many(adsorption_docs)
        print('[%s] Created %i new entries in the adsorption_%s collection'
              % (datetime.now(), len(adsorption_docs), dft_calculator))


def _find_atoms_docs_not_in_adsorption_collection(dft_calculator):
    '''
    This function will get the Mongo documents of adsorption calculations that
    are inside our `atoms` collection, but not inside our `adsorption`
    collection.

    Arg:
        dft_calculator  A string indicating which DFT calculator you want to
                        parse data for---e.g., 'vasp', 'qe', or 'rism'. This
                        function will find the data from the appropriate
                        collection.
    Returns:
        missing_ads_docs    A list of adsorption documents from the `atoms`
                            collection that have not yet been added to the
                            `adsorption` collection.
    '''
    # Find the FWIDs of the documents inside our adsorption collection
    with get_mongo_collection('adsorption_' + dft_calculator) as collection:
        docs_adsorption = list(collection.find({}, {'fwids': 'fwids', '_id': 0}))
    fwids_in_adsorption = {doc['fwids']['slab+adsorbate'] for doc in docs_adsorption}

    # Find the FWIDs of the documents inside our atoms collection
    with get_mongo_collection('atoms') as collection:
        query = {'fwname.calculation_type': 'slab+adsorbate optimization',
                 'fwname.dft_settings._calculator': dft_calculator,
                 'fwname.adsorbate': {'$ne': ''}}
        projection = {'fwid': 'fwid', '_id': 0}
        docs_atoms = list(collection.find(query, projection))
        fwids_in_atoms = {doc['fwid'] for doc in docs_atoms}

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

    # Use the appropriate adsorption energy calculator
    dft_calculator = atoms_doc['fwname']['dft_settings']['_calculator']
    if dft_calculator == 'vasp' or 'qe':
        adsorption_calculator = CalculateAdsorptionEnergy
    elif dft_calculator == 'rism':
        adsorption_calculator = CalculateRismAdsorptionEnergy
    else:
        raise ValueError('"%s" is an unrecognized DFT calculator' % dft_calculator)

    # Use the correct gas phase settings
    gas_dft_settings = gas_settings()[dft_calculator]
    if dft_calculator == 'rism':
        for key in ['anion_concs', 'cation_concs']:
            gas_dft_settings[key] = atoms_doc['fwname']['dft_settings'][key]

    # Create, run, and return the output of the task
    task = adsorption_calculator(adsorption_site=adsorption_site,
                                 shift=atoms_doc['fwname']['shift'],
                                 top=atoms_doc['fwname']['top'],
                                 adsorbate_name=atoms_doc['fwname']['adsorbate'],
                                 rotation=atoms_doc['fwname']['adsorbate_rotation'],
                                 mpid=atoms_doc['fwname']['mpid'],
                                 miller_indices=atoms_doc['fwname']['miller'],
                                 bare_slab_dft_settings=atoms_doc['fwname']['dft_settings'],
                                 adslab_dft_settings=atoms_doc['fwname']['dft_settings'],
                                 gas_dft_settings=gas_dft_settings)
    try:
        run_task(task)
        energy_doc = get_task_output(task)
        return energy_doc

    # If a task has failed and not produced an output, we don't want that to
    # stop us from updating the successful runs.
    except FileNotFoundError:
        pass

    # If the output already exists, then load and return it
    except luigi.target.FileAlreadyExists:
        energy_doc = get_task_output(task)
        return energy_doc

    # If some other error pops up, then we want to report it. But we also want
    # to move on so that we can still update other things.
    except:     # noqa: E722
        traceback.print_exc()
        doc_str = pprint.pformat({'fwname': atoms_doc['fwname'],
                                  'fwid': atoms_doc['fwid'],
                                  'directory': atoms_doc['directory'],
                                  'calculation_date': atoms_doc['calculation_date']})
        warnings.warn('We caught the exception reported just above and moved on '
                      'without updating the adsorption collection. Here is the '
                      'offending document:\n%s' % doc_str)


def __clean_calc_energy_docs(docs, missing_docs):
    '''
    If a calculation fails, our helper function will return `None`. Let's
    take those out here.

    Also, sometimes we have duplicate calculations. When we go to calculate the
    adsorption energy of the older FWID, GASpy will pull the information of the
    newer FWID instead. So we'll just keep redumping newer FWID duplicates.
    This function will parse out any repeat documents to avoid this.

    Arg:
        docs            A list of dictionaries obtained from the output of the
                        `CalculateAdsorptionEnergy` task
        missing_docs    The output of the
                        `_find_atoms_docs_not_in_adsorption_collection`
    Returns:
        cleaned_docs    The `docs` arguments, but with empty and duplicate
                        documents removed
    '''
    missing_fwids = {doc['fwid'] for doc in missing_docs}
    cleaned_docs = []
    for doc in docs:
        if doc is not None:
            fwid = doc['fwids']['adslab']
            if fwid in missing_fwids:
                cleaned_docs.append(doc)
                missing_fwids.remove(fwid)
    return cleaned_docs


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
    adsorption_doc['dft_settings'] = adslab_doc['fwname']['dft_settings']
    adsorption_doc['fwids'] = energy_doc['fwids']
    adsorption_doc['fw_directories'] = {'slab+adsorbate': adslab_doc['directory'],
                                        'slab': slab_doc['directory']}
    adsorption_doc['fp_final'] = fp_final
    adsorption_doc['fp_init'] = fp_init
    adsorption_doc['movement_data'] = {'max_bare_slab_movement': max_bare_slab_movement,
                                       'max_slab_movement': max_slab_movement,
                                       'max_adsorbate_movement': max_ads_movement}
    adsorption_doc['calculation_dates'] = {'slab+adsorbate': adslab_doc['calculation_date'],
                                           'slab': slab_doc['calculation_date']}
    return adsorption_doc
