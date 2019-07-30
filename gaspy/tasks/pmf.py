'''
This submodule houses some development tasks we use for Potential Mean Force
calculations. They are not unit tested and will not be stable; they will
probably not have their results stored in GASdb; and they may require some
GASpy expertise to use correctly. Use with caution.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pickle
import numpy as np
import luigi
from .core import make_task_output_object, save_task_output, get_task_output
from .calculation_finders import FindCalculation
from .atoms_generators import GenerateAdsorptionSites
from .. import defaults
from ..utils import unfreeze_dict
from ..mongo import make_atoms_from_doc, make_doc_from_atoms
from ..atoms_operatiors import add_adsorbate_onto_slab
from ..fireworks_helper_scripts import make_firework, submit_fwork
from ..make_fireworks import MakeAdslabFW

GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()
MAX_FIZZLES = defaults.MAX_FIZZLES


class CalculatePMF(luigi.Task):
    '''
    This task will calculate the Potential Mean Force (PMF) for you.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        adsorbate_heights   A list of floats indicating the different heights at
                            which you want to put the adsorbate (in Angstroms)
        adsorption_site     A 3-tuple of floats containing the Cartesian
                            coordinates of the adsorption site you want to
                            make a FW for
        mpid                A string indicating the Materials Project ID of
                            the bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices
                            of the slab[s] you want to enumerate sites from
        shift               A float indicating the shift of the slab
        top                 A Boolean indicating whether the adsorption
                            site is on the top or the bottom of the slab
        dft_settings        A dictionary containing your DFT settings
                            for the adslab relaxation
        bulk_dft_settings   A dictionary containing the DFT settings of
                            the relaxed bulk to enumerate slabs from
        max_fizzles         The maximum number of times you want any single
                            DFT calculation to fail before giving up on this.
    Returns:
        docs    A list of dictionaries with the following keys:
                adsorption_energy   A float indicating the adsorption energy
                fwids               A subdictionary whose keys are 'adslab' and
                                    'slab', and whose values are the FireWork
                                    IDs of the respective calculations.
    '''
    adsorbate = luigi.DictParameter()
    adsorbate_heights = luigi.ListParameter()
    adsorption_site = luigi.TupleParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS['rism'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['qe'])
    max_fizzles = luigi.IntParameter(MAX_FIZZLES)

    def requires(self):
        dependencies = {}

        # Find adsorbate+slab systems for each height
        for height in self.adsorbate_heights:
            dependencies[height] = FindFrozenAdslab(adsorbate=self.adsorbate,
                                                    adsorbate_height=height,
                                                    adsorption_site=self.adsorption_site,
                                                    mpid=self.mpid,
                                                    miller_indices=self.miller_indices,
                                                    shift=self.shift,
                                                    top=self.top,
                                                    dft_settings=self.dft_settings,
                                                    bulk_dft_settings=self.bulk_dft_settings,
                                                    max_fizzles=self.max_fizzles)

        # Make sure that we also grab the relaxed system, which we will assign
        # the height change of "0" Angstroms
        dependencies[0.] = FindRelaxedAdslab(adsorbate=self.adsorbate,
                                             adsorption_site=self.adsorption_site,
                                             mpid=self.mpid,
                                             miller_indices=self.miller_indices,
                                             shift=self.shift,
                                             top=self.top,
                                             dft_settings=self.dft_settings,
                                             bulk_dft_settings=self.bulk_dft_settings,
                                             max_fizzles=self.max_fizzles)
        return dependencies

    def run(self):
        '''
        Tell Luigi to calculate the relative energies of the different frozen
        adsorbate+slab systems
        '''
        # Parse the requirements to get all of the potential energies of the
        # different systems
        energies_by_height = {}
        for height, task in self.requires().items():
            doc = get_task_output(task)
            atoms = make_atoms_from_doc(doc)
            energy = atoms.get_potential_energy(apply_constraints=False)
            energies_by_height[height] = energy

        # Calculate the relative energies by subtracting off the energy of the
        # system where the adsorbate height is not changed
        base_energy = energies_by_height[0.]
        relative_energies = {height: energy - base_energy
                             for height, energy in energies_by_height.items()}
        save_task_output(self, relative_energies)

    def output(self):
        make_task_output_object(self)

    def get_relative_energies(self):
        '''
        This method will give you the relative energies

        Returns:
            relative_energies   A dictionary whose keys are the adjusted
                                heights (in Angstroms) of each adsorbate in the
                                various frozen structures, and whose values are
                                their energies (in eV) relative to the base
                                case, where the height change was 0 Angstroms.
        '''
        return get_task_output(self)


class FindRelaxedAdslab(FindCalculation):
    '''
    This task will find/submit will take any adsorbate you give it and then
    relax it onto a given adsorption site for you.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        adsorption_site     A 3-tuple of floats containing the Cartesian
                            coordinates of the adsorption site you want to make
                            a FW for.
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices of
                            the slab[s] you want to enumerate sites from
        shift               A float indicating the shift of the slab
        top                 A Boolean indicating whether the adsorption site is
                            on the top or the bottom of the slab
        dft_settings        A dictionary of the Adslab DFT settings you want to
                            use
        bulk_dft_settings   A dictionary containing the DFT settings of the
                            relaxed bulk to enumerate slabs from
        max_fizzles         The maximum number of times you want any single DFT
                            calculation to fail before giving up on this.
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    adsorbate = luigi.Parameter()
    adsorption_site = luigi.TupleParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS['rism'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['qe'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.adsorbate_atoms = make_atoms_from_doc(self.adsorbate)

        self.gasdb_query = {'fwname.calculation_type': 'PMF relaxation',
                            'fwname.adsorbate': self.adsorbate,
                            'fwname.adsorption_site.0': {'$gte': self.adsorption_site[0] - 1e-2,
                                                         '$lte': self.adsorption_site[0] + 1e-2},
                            'fwname.adsorption_site.1': {'$gte': self.adsorption_site[1] - 1e-2,
                                                         '$lte': self.adsorption_site[1] + 1e-2},
                            'fwname.adsorption_site.2': {'$gte': self.adsorption_site[2] - 1e-2,
                                                         '$lte': self.adsorption_site[2] + 1e-2},
                            'fwname.mpid': self.mpid,
                            'fwname.miller': self.miller_indices,
                            'fwname.shift': {'$gte': self.shift - 1e-3,
                                             '$lte': self.shift + 1e-4},
                            'fwname.top': self.top}

        self.fw_query = {'name.calculation_type': 'PMF relaxation',
                         'name.adsorbate': self.adsorbate,
                         'name.adsorption_site.0': {'$gte': self.adsorption_site[0] - 1e-2,
                                                    '$lte': self.adsorption_site[0] + 1e-2},
                         'name.adsorption_site.1': {'$gte': self.adsorption_site[1] - 1e-2,
                                                    '$lte': self.adsorption_site[1] + 1e-2},
                         'name.adsorption_site.2': {'$gte': self.adsorption_site[2] - 1e-2,
                                                    '$lte': self.adsorption_site[2] + 1e-2},
                         'name.mpid': self.mpid,
                         'name.miller': self.miller_indices,
                         'name.shift': {'$gte': self.shift - 1e-3,
                                        '$lte': self.shift + 1e-3},
                         'name.top': self.top}

        for key, value in self.dft_settings.items():
            self.gasdb_query['fwname.dft_settings.%s' % key] = value
            self.fw_query['name.dft_settings.%s' % key] = value

        self.dependency = MakeRelaxedAdslabFW(adsorbate=self.adsorbate,
                                              adsorption_site=self.adsorption_site,
                                              mpid=self.mpid,
                                              miller_indices=self.miller_indices,
                                              shift=self.shift,
                                              top=self.top,
                                              dft_settings=self.dft_settings,
                                              bulk_dft_settings=self.bulk_dft_settings)


class MakeRelaxedAdslabFW(MakeAdslabFW):
    '''
    Creates and submits a PMF-type adslab relaxation.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        adsorption_site     A 3-tuple of floats containing the Cartesian
                            coordinates of the adsorption site you want to make
                            a FW for.
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices of
                            the slab[s] you want to enumerate sites from
        shift               A float indicating the shift of the slab
        top                 A Boolean indicating whether the adsorption site is
                            on the top or the bottom of the slab
        dft_settings        A dictionary of the Adslab DFT settings you want to
                            use
        bulk_dft_settings   A dictionary containing the DFT settings of the
                            relaxed bulk to enumerate slabs from
    saved output:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about the sites. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These objects have a the adsorbate
                tagged with a `1`. These documents also contain the following
                fields:
                    fwids           A subdictionary containing the FWIDs of the
                                    prerequisite calculations
                    shift           Float indicating the shift/termination of
                                    the slab
                    top             Boolean indicating whether or not the slab
                                    is oriented upwards with respect to the way
                                    it was enumerated originally by pymatgen
                    slab_repeat     2-tuple of integers indicating the number
                                    of times the unit slab was repeated in the
                                    x and y directions before site enumeration
                    adsorption_site `np.ndarray` of length 3 containing the
                                    containing the cartesian coordinates of the
                                    adsorption site.
    '''
    adsorbate = luigi.DictParameter()
    adsorption_site = luigi.TupleParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS['rism'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['qe'])

    def requires(self):
        return GenerateUnrelaxedPMFAdslabs(adsorbate=self.adsorbate,
                                           mpid=self.mpid,
                                           miller_indices=self.miller_indices,
                                           bulk_dft_settings=self.bulk_dft_settings)

    def run(self):
        '''
        Override the parent class' `run` method with one designed specifically
        for PMF relaxations
        '''
        # Parse the possible adslab structures and find the one that matches
        # the site, shift, and top values we're looking for
        with open(self.input().path, 'rb') as file_handle:
            adslab_docs = pickle.load(file_handle)
        doc = self._find_matching_adslab_doc(adslab_docs=adslab_docs,
                                             adsorption_site=self.adsorption_site,
                                             shift=self.shift,
                                             top=self.top)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'PMF relaxation',
                   'adsorbate': self.adsorbate,
                   'adsorption_site': self.adsorption_site,
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'top': self.top,
                   'slab_repeat': doc['slab_repeat'],
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork)    # noqa: F841

        # Let Luigi know that we've made the FireWork
        self._complete = True


class GenerateUnrelaxedPMFAdslabs(luigi.Task):
    '''
    This class takes a set of adsorbate positions from the
    `GenerateAdsorptionSites` task and replaces the markers (uranium atoms)
    with the adsorbate you give it.

    Note that this is nearly identical to the
    `gaspy.tasks.atoms_generators.GenerateAdslabs` task, except this one has
    less optional arguments. It is therefore less flexible, but easier to use
    and less likely to break.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        mpid                A string indicating the Materials Project ID of
                            the bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices
                            of the slab[s] you want to enumerate sites from
        bulk_dft_settings   A dictionary containing the DFT settings of
                            the relaxed bulk to enumerate slabs from
    saved output:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about the sites. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These objects have a the adsorbate
                tagged with a `1`. These documents also contain the following
                fields:
                    fwids           A subdictionary containing the FWIDs of the
                                    prerequisite calculations
                    shift           Float indicating the shift/termination of
                                    the slab
                    top             Boolean indicating whether or not the slab
                                    is oriented upwards with respect to the way
                                    it was enumerated originally by pymatgen
                    slab_repeat     2-tuple of integers indicating the number
                                    of times the unit slab was repeated in the
                                    x and y directions before site enumeration
                    adsorption_site `np.ndarray` of length 3 containing the
                                    containing the cartesian coordinates of the
                                    adsorption site.
    '''
    adsorbate = luigi.DictParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['qe'])

    def requires(self):
        return GenerateAdsorptionSites(mpid=self.mpid,
                                       miller_indices=self.miller_indices,
                                       bulk_dft_settings=self.bulk_dft_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            site_docs = pickle.load(file_handle)

        # Fetch each slab and then replace the Uranium marker with the
        # adsorbate
        adsorbate = make_atoms_from_doc(self.adsorbate)
        docs_adslabs = []
        for site_doc in site_docs:
            slab = make_atoms_from_doc(site_doc)
            del slab[-1]
            adslab = add_adsorbate_onto_slab(adsorbate=adsorbate,
                                             slab=slab,
                                             site=site_doc['adsorption_site'])

            # Turn the adslab into a document, add the correct fields, and save
            doc = make_doc_from_atoms(adslab)
            doc['fwids'] = site_doc['fwids']
            doc['shift'] = site_doc['shift']
            doc['top'] = site_doc['top']
            doc['slab_repeat'] = site_doc['slab_repeat']
            doc['adsorption_site'] = site_doc['adsorption_site']
            docs_adslabs.append(doc)
        save_task_output(self, docs_adslabs)

    def output(self):
        return make_task_output_object(self)


class FindFrozenAdslab(FindRelaxedAdslab):
    '''
    This task will find/submit will take any adsorbate you give it and then
    relax it onto a given adsorption site for you.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        adsorbate_height    A float indicating the height at which you want to
                            put the adsorbate
        adsorption_site     A 3-tuple of floats containing the Cartesian
                            coordinates of the adsorption site you want to make
                            a FW for.
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices of
                            the slab[s] you want to enumerate sites from
        shift               A float indicating the shift of the slab
        top                 A Boolean indicating whether the adsorption site is
                            on the top or the bottom of the slab
        dft_settings        A dictionary of the Adslab DFT settings you want to
                            use
        bulk_dft_settings   A dictionary containing the DFT settings of the
                            relaxed bulk to enumerate slabs from
        max_fizzles         The maximum number of times you want any single DFT
                            calculation to fail before giving up on this.
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    adsorbate_height = luigi.FloatParamete()

    def _load_attributes(self):
        '''
        This will be almost identical to finding a relaxed PMF adslab, but with
        a few of changes. So we just call the parent method and then modify the
        appropriate attributes.
        '''
        super()._load_attributes()

        # These calculations will be frozen, and they'll also have varying
        # adsorbate heights
        self.gasdb_query['fwname.calculation_type'] = 'PMF frozen scf'
        self.gasdb_query['fwname.adsorbate_height'] = self.adsorbate_height
        self.fw_query['name.calculation_type'] = 'PMF frozen scf'
        self.fw_query['name.adsorbate_height'] = self.adsorbate_height

        # Change the dependency to the frozen FW, not the relaxed one
        self.dependency = MakeFrozenAdslabFW(adsorbate=self.adsorbate,
                                             adsorbate_height=self.adsorbate_height,
                                             adsorption_site=self.adsorption_site,
                                             mpid=self.mpid,
                                             miller_indices=self.miller_indices,
                                             shift=self.shift,
                                             top=self.top,
                                             dft_settings=self.dft_settings,
                                             bulk_dft_settings=self.bulk_dft_settings)


class MakeFrozenAdslabFW(luigi.Task):
    '''
    Creates and submits a PMF-type structure where we change the height of the
    adsorbate and freeze the atoms.

    Args:
        adsorbate           Use `make_adsorbate_dict` in this submodule to turn
                            an `ase.Atoms` object into a dictionary, then pass
                            that dictionary here.
        adsorbate_height    A float indicating the height at which you want to
                            put the adsorbate
        adsorption_site     A 3-tuple of floats containing the Cartesian
                            coordinates of the adsorption site you want to make
                            a FW for.
        mpid                A string indicating the Materials Project ID of the
                            bulk you want to enumerate sites from
        miller_indices      A 3-tuple containing the three Miller indices of
                            the slab[s] you want to enumerate sites from
        shift               A float indicating the shift of the slab
        top                 A Boolean indicating whether the adsorption site is
                            on the top or the bottom of the slab
        dft_settings        A dictionary of the Adslab DFT settings you want to
                            use
        bulk_dft_settings   A dictionary containing the DFT settings of the
                            relaxed bulk to enumerate slabs from
    saved output:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about the sites. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These objects have a the adsorbate
                tagged with a `1`. These documents also contain the following
                fields:
                    fwids           A subdictionary containing the FWIDs of the
                                    prerequisite calculations
                    shift           Float indicating the shift/termination of
                                    the slab
                    top             Boolean indicating whether or not the slab
                                    is oriented upwards with respect to the way
                                    it was enumerated originally by pymatgen
                    slab_repeat     2-tuple of integers indicating the number
                                    of times the unit slab was repeated in the
                                    x and y directions before site enumeration
                    adsorption_site `np.ndarray` of length 3 containing the
                                    containing the cartesian coordinates of the
                                    adsorption site.
    '''
    adsorbate = luigi.DictParameter()
    adsorbate_height = luigi.FloatParamete()
    adsorption_site = luigi.TupleParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS['rism'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['qe'])

    def requires(self):
        return FindRelaxedAdslab(adsorbate=self.adsorbate,
                                 adsorption_site=self.adsorption_site,
                                 mpid=self.mpid,
                                 miller_indices=self.miller_indices,
                                 shift=self.shift,
                                 top=self.top,
                                 dft_settings=self.dft_settings,
                                 bulk_dft_settings=self.bulk_dft_settings,
                                 max_fizzles=self.max_fizzles)

    def run(self):
        '''
        Override the parent class' `run` method with one designed specifically
        for PMF relaxations
        '''
        self._parse_input()
        self._move_adsorbate()
        self._submit_fw()
        self._complete = True

    def _parse_input(self):
        '''
        Parse the possible adslab structures and find the one that matches
        the site, shift, and top values we're looking for
        '''
        with open(self.input().path, 'rb') as file_handle:
            adslab_docs = pickle.load(file_handle)
        doc = self._find_matching_adslab_doc(adslab_docs=adslab_docs,
                                             adsorption_site=self.adsorption_site,
                                             shift=self.shift,
                                             top=self.top)
        self.relaxed_adslab_doc = doc
        self.atoms = make_atoms_from_doc(self.relaxed_adslab_doc)

    def _move_adsorbate(self, atoms):
        '''
        Move the adsorbate up or down according to the prescribed height
        '''
        translation = np.empty((len(atoms), 3))
        for i, atom in enumerate(atoms):
            if atom.tag > 0:
                translation[i, :] = np.array([0., 0., self.adsorbate_height])
        atoms.translate(translation)

    def _submit_fw(self):
        '''
        Create, package, and submit the FireWork
        '''
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'PMF frozen scf',
                   'adsorbate': self.adsorbate,
                   'adsorbate_height': self.adsorbate_height,
                   'adsorption_site': self.adsorption_site,
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'top': self.top,
                   'slab_repeat': self.relaxed_adslab_doc['slab_repeat'],
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=self.atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork)    # noqa: F841


def make_adsorbate_dict(atoms):
    '''
    This is a light wrapper around the `make_atoms_from_doc` function, but it
    takes out some fields that we don't want to pass to Luigi.

    Arg:
        atoms   ase.Atoms object
    Returns:
        doc     A dictionary with the standard subdocuments: atoms, calculator,
                results
    '''
    doc = make_doc_from_atoms(atoms)
    for key in ['user', 'ctime', 'mtime']:
        doc.pop(key)
    return doc
