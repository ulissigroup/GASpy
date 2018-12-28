'''
This submodule contains various tasks that finds calculations of various atoms
structures by looking through our database.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import luigi
from .. import defaults
from ..gasdb import get_mongo_collection
from ..fireworks_helper_scripts import is_rocket_running
from .core import save_task_output, make_task_output_object

GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS


class FindGas(luigi.Task):
    gas_name = luigi.Parameter()
    vasp_settings = luigi.DictParameter()

    def run(self):
        # Parse the task's input search parameters into a Mongo-readable query,
        # which we can then use to find the Mongo document[s] of the
        # relaxation[s] we're looking for
        query = {'type': 'gas', 'fwname.gasname': self.gas_name}
        docs = _find_docs_in_atoms_collection(query, self.vasp_settings)

        # If we have any matches, then remove the old ones and save the newest
        try:
            doc = _remove_old_docs(docs)
            save_task_output(self, doc)

        # If there's no match in our `atoms` collection, then check if our
        # FireWorks system is currently running it
        except SyntaxError:
            fw_query = {'name': 'gas phase optimization',
                        'name.gasname': self.gas_name}
            # If we're not running, then submit the job
            if is_rocket_running(fw_query, self.vasp_settings) is False:
                return 'placeholder for "we finished but it is not in our db yet'
            # If we are running, then just wait
            else:
                pass

    def output(self):
        return make_task_output_object(self)


def _find_docs_in_atoms_collection(query, vasp_settings):
    '''
    This function will search our "atoms" Mongo collection using whatever Mongo
    query you specify. It will also parse out the 'vasp_settings' part of the
    query for you.

    Args:
        query           A dictionary that can be fed to the `query' argument of
                        a Mongo collection's `find` method.
        vasp_settings   A dictionary containing the various VASP settings used
                        for calculation.
    Returns:
        docs    A list of dictionaries, also known as Mongo documents, from
                our `atoms` Mongo collection that match your query
    '''
    for key, value in vasp_settings.items():
        query['fwname.vasp_settings.%s' % key] = value
    with get_mongo_collection('atoms') as collection:
        docs = list(collection.find(query))
    return docs


def _remove_old_docs(docs):
    '''
    This function will parse out Mongo documents that have older FireWork ID
    numbers and warn the user if it happens.

    Arg:
        docs    A list of dictionaries that should have the key 'fw_id' in them
    Returns:
        doc     The one document/dictionary in the input that has the highest
                'fw_id' value
    '''
    if len(docs) == 1:
        return docs[0]

    # Warn the user if they have more than one match, then pass the newest one
    elif len(docs) > 1:
        docs = sorted(docs, key=lambda doc: doc['fwid'], reverse=True)
        doc_latest = docs[0]
        fwids = [str(doc['fwid']) for doc in docs]
        message = ('These completed FireWorks rockets look identical:  %s. '
                   'We will be using the latest one, %s'
                   % (', '.join(fwids), doc_latest['fwid']))
        warnings.warn(message, RuntimeWarning)
        return doc_latest

    elif len(docs) == 0:
        raise SyntaxError('You tried to parse out old documents, but did not '
                          'pass any documents at all.')


#class FindBulk(luigi.Task):
#        search_strings = {'type': 'bulk',
#                          'fwname.mpid': self.parameters['bulk']['mpid']}
#        for key in self.parameters['bulk']['vasp_settings']:
#            search_strings['fwname.vasp_settings.%s' % key] = \
#                self.parameters['bulk']['vasp_settings'][key]
#
#
#class FindSlab(luigi.Task):
#    elif self.calctype == 'slab':
#        search_strings = {'type': 'slab',
#                          'fwname.miller': list(self.parameters['slab']['miller']),
#                          'fwname.top': self.parameters['slab']['top'],
#                          'fwname.shift': self.parameters['slab']['shift'],
#                          'fwname.mpid': self.parameters['bulk']['mpid']}
#        for key in self.parameters['slab']['vasp_settings']:
#            if key not in ['isym']:
#                search_strings['fwname.vasp_settings.%s' % key] = \
#                    self.parameters['slab']['vasp_settings'][key]
#
#
#class FindSurface(luigi.Task):
#    elif self.calctype == 'slab_surface_energy':
#        # pretty much identical to "slab" above, except no top since top/bottom
#        # surfaces are both relaxed
#        search_strings = {'type': 'slab_surface_energy',
#                          'fwname.miller': list(self.parameters['slab']['miller']),
#                          'fwname.num_slab_atoms': self.parameters['slab']['natoms'],
#                          'fwname.shift': self.parameters['slab']['shift'],
#                          'fwname.mpid': self.parameters['bulk']['mpid']}
#        for key in self.parameters['slab']['vasp_settings']:
#            if key not in ['isym']:
#                search_strings['fwname.vasp_settings.%s' % key] = \
#                    self.parameters['slab']['vasp_settings'][key]
#
#
#class FindAdslab(luigi.Task):
#    elif self.calctype == 'slab+adsorbate':
#        search_strings = {'type': 'slab+adsorbate',
#                          'fwname.miller': list(self.parameters['slab']['miller']),
#                          'fwname.top': self.parameters['slab']['top'],
#                          'fwname.shift': self.parameters['slab']['shift'],
#                          'fwname.mpid': self.parameters['bulk']['mpid'],
#                          'fwname.adsorbate': self.parameters['adsorption']['adsorbates'][0]['name']}
#        for key in self.parameters['adsorption']['vasp_settings']:
#            if key not in ['nsw', 'isym', 'symprec']:
#                search_strings['fwname.vasp_settings.%s' % key] = \
#                    self.parameters['adsorption']['vasp_settings'][key]
#        if 'adsorption_site' in self.parameters['adsorption']['adsorbates'][0]:
#            search_strings['fwname.adsorption_site'] = \
#                self.parameters['adsorption']['adsorbates'][0]['adsorption_site']
#        if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
#            for key in self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation']:
#                search_strings['fwname.adsorbate_rotation.%s' % key] = \
#                    self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation'][key]
#
#    # Round the shift to 4 decimal places so that we will be able to match shift numbers
#    if 'fwname.shift' in search_strings:
#        shift = search_strings['fwname.shift']
#        search_strings['fwname.shift'] = {'$gte': shift - 1e-4, '$lte': shift + 1e-4}
#
#    # Grab all of the matching entries in the Auxiliary database
#    with gasdb.get_mongo_collection('atoms') as collection:
#        self.matching_doc = list(collection.find(search_strings))
#
#    # If there are no matching entries, we need to yield a requirement that will
#    # generate the necessary unrelaxed structure
#    if len(self.matching_doc) == 0:
#        if self.calctype == 'slab':
#            return [GenerateSlabs(OrderedDict(bulk=self.parameters['bulk'],
#                                              slab=self.parameters['slab'])),
#                    # We are also vaing the unrelaxed slabs just in case. We can delete if
#                    # we can find shifts successfully.
#                    GenerateSlabs(OrderedDict(unrelaxed=True,
#                                              bulk=self.parameters['bulk'],
#                                              slab=self.parameters['slab']))]
#        if self.calctype == 'slab_surface_energy':
#            return [GenerateSlabs(OrderedDict(bulk=self.parameters['bulk'],
#                                              slab=self.parameters['slab'])),
#                    # We are also vaing the unrelaxed slabs just in case. We can delete if
#                    # we can find shifts successfully.
#                    GenerateSlabs(OrderedDict(unrelaxed=True,
#                                              bulk=self.parameters['bulk'],
#                                              slab=self.parameters['slab']))]
#        if self.calctype == 'slab+adsorbate':
#            # Return the base structure, and all possible matching ones for the surface
#            search_strings = {'type': 'slab+adsorbate',
#                              'fwname.miller': list(self.parameters['slab']['miller']),
#                              'fwname.top': self.parameters['slab']['top'],
#                              'fwname.mpid': self.parameters['bulk']['mpid'],
#                              'fwname.adsorbate': self.parameters['adsorption']['adsorbates'][0]['name']}
#            with gasdb.get_mongo_collection('atoms') as collection:
#                self.matching_docs_all_calcs = list(collection.find(search_strings))
#
#            # If we don't modify the parameters, the parameters will contain the FP for the
#            # request. This will trigger a FingerprintUnrelaxedAdslabs for each FP, which
#            # is entirely unnecessary. The result is the same # regardless of what
#            # parameters['adsorption'][0]['fp'] happens to be
#            parameters_copy = utils.unfreeze_dict(copy.deepcopy(self.parameters))
#            if 'fp' in parameters_copy['adsorption']['adsorbates'][0]:
#                del parameters_copy['adsorption']['adsorbates'][0]['fp']
#            parameters_copy['unrelaxed'] = 'relaxed_bulk'
#            return FingerprintUnrelaxedAdslabs(parameters_copy)
#
#        if self.calctype == 'bulk':
#            return GenerateBulk({'bulk': self.parameters['bulk']})
#        if self.calctype == 'gas':
#            return GenerateGas({'gas': self.parameters['gas']})
