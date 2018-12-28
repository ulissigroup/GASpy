'''
This submodule contains various tasks that finds calculations of various atoms
structures by looking through our database.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import luigi
from .. import defaults
from .core import save_task_output, make_task_output_object
from .make_fireworks.core import MakeGasFW
from ..gasdb import get_mongo_collection
from ..fireworks_helper_scripts import is_rocket_running

GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS


class FindCalculation(luigi.Task):
    '''
    This is meant to be used as a parent class to the rest of the classes in
    this submodule. It contains the skeleton needed to find a calculation in
    our auxiliary Mongo database or our FireWorks database. If the calculation
    appears completed in our auxiliary Mongo database, then this task will
    return the results. If the calculation is pending, it will wait. If the
    calculation has not yet been submitted, then it will start the calculation.

    This class requires some attributes, Luigi arguments, and method
    definitions before it can be used.

    Required Luigi argument:
        vasp_settings   A dictionary containing the VASP settings of the
                        calculation you want to find. You can probably find
                        examples in `gaspy.defaults.*_SETTNGS`.
    Required attributes:
        gasdb_query     A dictionary that can be passed as a `query` argument
                        to our auxiliary Mongo database's `atoms` colection to
                        find a calculation
        fw_query        A dictionary that can be passed as a `query` argument
                        to our FireWorks database's `fireworks` collection to
                        find a calculation
        dependency      An instance of a `luigi.Task` child class that should
                        be returned as a dependency if the calculation is not
                        found
    Required method:
        _load_attributes    Some of the required attributes may require Luigi
                            parameters, which are only available after class
                            instantiation. This method should save these
                            attributes to the class. This method will be
                            called automatically at the start of `run`.
    '''
    def run(self, _testing=False):
        '''
        Arg:
            _testing    Boolean indicating whether or not you are doing a unit
                        test. You probably shouldn't touch this.
        '''
        self._load_attributes()

        # Find and save the calculations (if they're in our `atoms` collection)
        docs = _find_docs_in_atoms_collection(self.gasdb_query, self.vasp_settings)
        try:
            doc = _remove_old_docs(docs)
            save_task_output(self, doc)

        # If there's no match in our `atoms` collection, then check if our
        # FireWorks system is currently running it
        except SyntaxError:

            # If we're not running, then submit the job
            if is_rocket_running(self.fw_query,
                                 self.vasp_settings,
                                 _testing=_testing) is False:
                return self.dependency

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


class FindGas(FindCalculation):
    '''
    This task will try to find a gas phase calculation in either our auxiliary
    Mongo database or our FireWorks database. If the calculation is complete,
    then it will return the results. If the calculation is pending, it will
    wait. If the calculation has not yet been submitted, then it will start the
    calculation.

    Args:
        gas_name        A string indicating the name of the gas you are looking
                        for (e.g., 'CO')
        vasp_settings   A dictionary containing your VASP settings
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    gas_name = luigi.Parameter()
    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.gasdb_query = {'type': 'gas', 'fwname.gasname': self.gas_name}
        self.fw_query = {'name.calculation_type': 'gas phase optimization',
                         'name.gasname': self.gas_name}
        self.dependency = MakeGasFW(self.gas_name, self.vasp_settings)


#class FindBulk(FindCalculation):
#    '''
#    This task will try to find a unit cell calculation in either our auxiliary
#    Mongo database or our FireWorks database. If the calculation is complete,
#    then it will return the results. If the calculation is pending, it will
#    wait. If the calculation has not yet been submitted, then it will start the
#    calculation.
#
#    Args:
#        mpid            A string indicating the Materials Project ID of the
#                        bulk unit cell you are looking for
#        vasp_settings   A dictionary containing your VASP settings
#    Output:
#        doc     When the calculation is found in our auxiliary Mongo database
#                successfully, then this task's output will be a dictionary
#                containing information about the relaxed system. You should
#                also be able to turn the dictionary into an `ase.Atoms` object
#                using `gaspy.mongo.make_atoms_from_doc`.
#    '''
#    # Actual arguments for this task
#    mpid = luigi.Parameter()
#    vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
#
#    # Attributes required to make the parent task work
#    gasdb_query = {'type': 'bulk', 'fwname.mpid': mpid}
#    fw_query = {'name.calculation_type': 'unit cell optimization',
#                'name.mpid': mpid}
#    MakeBulkFW(mpid=mpid, vasp_settings=vasp_settings)


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
