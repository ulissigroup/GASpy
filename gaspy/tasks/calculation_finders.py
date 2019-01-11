'''
This submodule contains various tasks that finds calculations of various atoms
structures by looking through our database.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import luigi
from .. import defaults
from ..utils import turn_site_into_str
from ..gasdb import get_mongo_collection
from ..fireworks_helper_scripts import is_rocket_running
from .core import save_task_output, make_task_output_object
from .make_fireworks import (MakeGasFW,
                             MakeBulkFW,
                             MakeAdslabFW)

GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS
ADSLAB_SETTINGS = defaults.ADSLAB_SETTINGS


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
                        find a calculation that we have already done
        fw_query        A dictionary that can be passed as a `query` argument
                        to our FireWorks database's `fireworks` collection to
                        find a calculation that we may currently be doing
        dependency      An instance of a `luigi.Task` child class that should
                        be returned as a dependency if the calculation is not
                        found. Should probably be a task from
                        `gaspy.tasks.make_fireworks`.
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
        with get_mongo_collection('atoms') as collection:
            docs = list(collection.find(self.gasdb_query))
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
        self.gasdb_query = {'type': 'gas',
                            'fwname.gasname': self.gas_name}
        self.fw_query = {'name.calculation_type': 'gas phase optimization',
                         'name.gasname': self.gas_name}
        for key, value in self.vasp_settings.items():
            self.gasdb_query['fwname.vasp_settings.%s' % key] = value
            self.fw_query['name.vasp_settings.%s' % key] = value
        self.dependency = MakeGasFW(self.gas_name, self.vasp_settings)


class FindBulk(FindCalculation):
    '''
    This task will try to find a bulk calculation in either our auxiliary Mongo
    database or our FireWorks database. If the calculation is complete, then it
    will return the results. If the calculation is pending, it will wait. If
    the calculation has not yet been submitted, then it will start the
    calculation.

    Args:
        mpid            A string indicating the Materials Project ID of the bulk
                        you are looking for (e.g., 'mp-30')
        vasp_settings   A dictionary containing your VASP settings
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    mpid = luigi.Parameter()
    vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.gasdb_query = {'type': 'bulk',
                            'fwname.mpid': self.mpid}
        self.fw_query = {'name.calculation_type': 'unit cell optimization',
                         'name.mpid': self.mpid}
        for key, value in self.vasp_settings.items():
            self.gasdb_query['fwname.vasp_settings.%s' % key] = value
            self.fw_query['name.vasp_settings.%s' % key] = value
        self.dependency = MakeBulkFW(self.mpid, self.vasp_settings)


class FindAdslab(FindCalculation):
    '''
    This task will try to find a bulk calculation in either our auxiliary Mongo
    database or our FireWorks database. If the calculation is complete, then it
    will return the results. If the calculation is pending, it will wait. If
    the calculation has not yet been submitted, then it will start the
    calculation.

    Args:
        adsorption_site         A 3-tuple of floats containing the Cartesian
                                coordinates of the adsorption site you want to
                                make a FW for
        shift                   A float indicating the shift of the slab
        top                     A Boolean indicating whether the adsorption
                                site is on the top or the bottom of the slab
        vasp_settings           A dictionary containing your VASP settings
                                for the adslab relaxation
        adsorbate_name          A string indicating which adsorbate to use. It
                                should be one of the keys within the
                                `gaspy.defaults.ADSORBATES` dictionary. If you
                                want an adsorbate that is not in the dictionary,
                                then you will need to add the adsorbate to that
                                dictionary.
        rotation                A dictionary containing the angles (in degrees)
                                in which to rotate the adsorbate after it is
                                placed at the adsorption site. The keys for
                                each of the angles are 'phi', 'theta', and
                                psi'.
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to enumerate sites from
        miller_indices          A 3-tuple containing the three Miller indices
                                of the slab[s] you want to enumerate sites from
        min_xy                  A float indicating the minimum width (in both
                                the x and y directions) of the slab (Angstroms)
                                before we enumerate adsorption sites on it.
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here
                                as a dictionary.
        bulk_vasp_settings      A dictionary containing the VASP settings of
                                the relaxed bulk to enumerate slabs from
        mpid            A string indicating the Materials Project ID of the bulk
                        you are looking for (e.g., 'mp-30')
        vasp_settings   A dictionary containing your VASP settings
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    adsorption_site = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    vasp_settings = luigi.DictParameter(ADSLAB_SETTINGS['vasp'])
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.gasdb_query = {'type': 'slab+adsorbate',
                            'fwname.adsorbate': self.adsorbate_name,
                            'fwname.adsorption_site': turn_site_into_str(self.adsorption_site),
                            'fwname.mpid': self.mpid,
                            'fwname.miller': self.miller_indices,
                            'fwname.shift': {'$gte': self.shift - 1e-4,
                                             '$lte': self.shift + 1e-4},
                            'fwname.top': self.top,
                            'fwname.adsorbate_rotation.phi': self.rotation['phi'],
                            'fwname.adsorbate_rotation.theta': self.rotation['theta'],
                            'fwname.adsorbate_rotation.psi': self.rotation['psi']}
        self.fw_query = {'name.calculation_type': 'slab+adsorbate optimization',
                         'name.adsorbate': self.adsorbate_name,
                         'name.adsorption_site': turn_site_into_str(self.adsorption_site),
                         'name.mpid': self.mpid,
                         'name.miller': self.miller_indices,
                         'name.shift': {'$gte': self.shift - 1e-4,
                                        '$lte': self.shift + 1e-4},
                         'name.top': self.top,
                         'name.adsorbate_rotation.phi': self.rotation['phi'],
                         'name.adsorbate_rotation.theta': self.rotation['theta'],
                         'name.adsorbate_rotation.psi': self.rotation['psi']}

        for key, value in self.vasp_settings.items():
            # We don't care if these VASP settings change
            if key not in set(['nsw', 'isym', 'symprec']):
                self.gasdb_query['fwname.vasp_settings.%s' % key] = value
                self.fw_query['name.vasp_settings.%s' % key] = value

        # For historical reasons, we do bare slab relaxations with the adslab
        # infrastructure. If this task happens to be for a bare slab, then we
        # should take out some extraneous adsorbate information. We should have
        # a separate set of tasks, but that's for future us to fix.
        if self.adsorbate_name == '':
            del self.gasdb_query['fwname.adsorption_site']
            del self.gasdb_query['fwname.adsorbate_rotation.phi']
            del self.gasdb_query['fwname.adsorbate_rotation.theta']
            del self.gasdb_query['fwname.adsorbate_rotation.psi']
            del self.fw_query['name.adsorption_site']
            del self.fw_query['name.adsorbate_rotation.phi']
            del self.fw_query['name.adsorbate_rotation.theta']
            del self.fw_query['name.adsorbate_rotation.psi']

        self.dependency = MakeAdslabFW(adsorption_site=self.adsorption_site,
                                       shift=self.shift,
                                       top=self.top,
                                       vasp_settings=self.vasp_settings,
                                       adsorbate_name=self.adsorbate_name,
                                       rotation=self.rotation,
                                       mpid=self.mpid,
                                       miller_indices=self.miller_indices,
                                       min_xy=self.min_xy,
                                       slab_generator_settings=self.slab_generator_settings,
                                       get_slab_settings=self.get_slab_settings,
                                       bulk_vasp_settings=self.bulk_vasp_settings)


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
