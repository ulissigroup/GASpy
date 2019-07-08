'''
This submodule contains various tasks that finds calculations of various atoms
structures by looking through our database.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
from copy import deepcopy
import pickle
import numpy as np
import luigi
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from .. import defaults
from ..mongo import make_atoms_from_doc, make_doc_from_atoms
from ..gasdb import get_mongo_collection
from ..fireworks_helper_scripts import find_n_rockets
from .core import save_task_output, make_task_output_object, get_task_output
from .atoms_generators import GenerateBulk
from .make_fireworks import (MakeGasFW,
                             MakeBulkFW,
                             MakeAdslabFW,
                             MakeSurfaceFW)

GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


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
        dft_settings    A dictionary containing the DFT settings of the
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
    max_fizzles = luigi.IntParameter(5)

    def run(self, _testing=False):
        '''
        Arg:
            _testing    Boolean indicating whether or not you are doing a unit
                        test. You probably shouldn't touch this.
        '''
        calc_found = self._find_and_save_calculation()

        # If there's no match in our `atoms` collection, then check if our
        # FireWorks system is currently running it
        if calc_found is False:
            n_running, n_fizzles = find_n_rockets(self.fw_query,
                                                  self.dft_settings,
                                                  _testing=_testing)

            # If we aren't running yet, then start running
            if n_running == 0:
                if n_fizzles < self.max_fizzles:
                    yield self.dependency

                # If we've failed too often, then don't bother running.
                else:
                    raise ValueError('Since we have fizzled a calculation %i '
                                     'times, which is more than the '
                                     'specified threshold of %i, we will '
                                     'not be submitting this calculation '
                                     'again.'
                                     % (n_fizzles, self.max_fizzles))

            # If we're already running, then just move on
            else:
                pass

    def _find_and_save_calculation(self):
        '''
        Find and save the calculations (if they're in our `atoms` collection)

        Returns:
            bool    Boolean indicating whether or not we've successfully found
                    and saved the calculation
        '''
        # Find the document in our `atoms` Mongo collection
        try:
            with get_mongo_collection('atoms') as collection:
                docs = list(collection.find(self.gasdb_query))

        # If we have not yet created the query yet, then load it. We try/except
        # this so that we only load it once.
        except AttributeError:
            self._load_attributes()
            with get_mongo_collection('atoms') as collection:
                docs = list(collection.find(self.gasdb_query))

        # Save the match
        doc = self._remove_old_docs(docs)
        if doc != {}:
            try:
                save_task_output(self, doc)
            # If we've already saved the output, then move on
            except luigi.target.FileAlreadyExists:
                pass
            return True

        # If there is no match, then tell us so
        else:
            return False

    @staticmethod
    def _remove_old_docs(docs):
        '''
        This method will parse out Mongo documents that have older FireWork ID
        numbers and warn the user if it happens.

        Arg:
            docs    A list of dictionaries that should have the 'fw_id' key
        Returns:
            doc     The one document/dictionary in the input that has the
                    highest 'fw_id' value
        '''
        # If there's one document, just pass it along
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
        # If there's nothing, then just give an empty document back
        else:
            return {}

    def complete(self):
        '''
        This task is done when we can find a calculation. We make a custom
        `complete` method because we don't throw an error when we can't find
        something, and if we don't throw any errors, then Luigi will think that
        we're actually done.
        '''
        # If we can get the output from the pickles, then we're done
        try:
            _ = get_task_output(self)   # noqa: F841
            return True

        # If it's not pickled, then check Mongo
        except FileNotFoundError:
            return self._find_and_save_calculation()

    def output(self):
        return make_task_output_object(self)


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
        dft_settings    A dictionary containing your DFT settings
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    gas_name = luigi.Parameter()
    dft_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.gasdb_query = {'fwname.calculation_type': 'gas phase optimization',
                            'fwname.gasname': self.gas_name}
        self.fw_query = {'name.calculation_type': 'gas phase optimization',
                         'name.gasname': self.gas_name}
        for key, value in self.dft_settings.items():
            self.gasdb_query['fwname.dft_settings.%s' % key] = value
            self.fw_query['name.dft_settings.%s' % key] = value
        self.dependency = MakeGasFW(self.gas_name, self.dft_settings)


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
        dft_settings    A dictionary containing your DFT settings
        k_pts_x         The number of k-points you want in the x-direction. It
                        is only used if `dft_settings['kpts'] == 'bulk'`.
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname',
                or 'results'. This document should  also be able to be turned
                an `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    mpid = luigi.Parameter()
    dft_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
    k_pts_x = luigi.IntParameter(10)

    def requires(self):
        '''
        This calculation finder is different because we need to create the
        `ase.Atoms` objects BEFORE performing the queries, because the queries
        will contain the number of atoms. And we don't know the number of atoms
        until we make the `ase.Atoms` object.
        '''
        return GenerateBulk(mpid=self.mpid)

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        # If the k-points is 'bulk', then calculate them
        if self.dft_settings['kpts'] == 'bulk':
            bulk_doc = get_task_output(self.requires())
            bulk_atoms = make_atoms_from_doc(bulk_doc)
            kpts = self.calculate_bulk_k_points(bulk_atoms, self.k_pts_x)
            self.dft_settings['kpts'] = kpts

        # Initialize the required attributes
        self.gasdb_query = {"fwname.calculation_type": "unit cell optimization",
                            "fwname.mpid": self.mpid}
        self.fw_query = {'name.calculation_type': 'unit cell optimization',
                         'name.mpid': self.mpid}
        for key, value in self.dft_settings.items():
            self.gasdb_query['fwname.dft_settings.%s' % key] = value
            self.fw_query['name.dft_settings.%s' % key] = value
        self.dependency = MakeBulkFW(self.mpid, self.dft_settings)

    @staticmethod
    def calculate_bulk_k_points(atoms, k_pts_x=10):
        '''
        For unit cell calculations, it's a good practice to calculate the
        k-point mesh given the unit cell size. We do that on-the-spot here.

        Args:
            atoms       The `ase.Atoms` object you want to relax
            k_pts_x     An integer indicating the number of k points you want in
                        the x-direction
        Returns:
            k_pts   A 3-tuple of integers indicating the k-point mesh to use
        '''
        cell = atoms.get_cell()
        a0 = np.linalg.norm(cell[0])
        b0 = np.linalg.norm(cell[1])
        c0 = np.linalg.norm(cell[2])
        k_pts = (k_pts_x,
                 max(1, int(k_pts_x*a0/b0)),
                 max(1, int(k_pts_x*a0/c0)))
        return k_pts


class FindAdslab(FindCalculation):
    '''
    This task will try to find an adsorption calculation in either our
    auxiliary Mongo database or our FireWorks database. If the calculation is
    complete, then it will return the results. If the calculation is pending,
    it will wait. If the calculation has not yet been submitted, then it will
    start the calculation.

    Args:
        adsorption_site         A 3-tuple of floats containing the Cartesian
                                coordinates of the adsorption site you want to
                                make a FW for
        shift                   A float indicating the shift of the slab
        top                     A Boolean indicating whether the adsorption
                                site is on the top or the bottom of the slab
        dft_settings            A dictionary containing your DFT settings
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
        bulk_dft_settings       A dictionary containing the DFT settings of
                                the relaxed bulk to enumerate slabs from
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
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS['vasp'])
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        self.gasdb_query = {'fwname.calculation_type': 'slab+adsorbate optimization',
                            'fwname.adsorbate': self.adsorbate_name,
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
                            'fwname.top': self.top,
                            'fwname.adsorbate_rotation.phi': self.rotation['phi'],
                            'fwname.adsorbate_rotation.theta': self.rotation['theta'],
                            'fwname.adsorbate_rotation.psi': self.rotation['psi']}
        self.fw_query = {'name.calculation_type': 'slab+adsorbate optimization',
                         'name.adsorbate': self.adsorbate_name,
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
                         'name.top': self.top,
                         'name.adsorbate_rotation.phi': self.rotation['phi'],
                         'name.adsorbate_rotation.theta': self.rotation['theta'],
                         'name.adsorbate_rotation.psi': self.rotation['psi']}

        for key, value in self.dft_settings.items():
            # We don't care if these VASP-DFT settings change
            if key not in set(['nsw', 'isym', 'symprec']):
                self.gasdb_query['fwname.dft_settings.%s' % key] = value
                self.fw_query['name.dft_settings.%s' % key] = value

        # For historical reasons, we do bare slab relaxations with the adslab
        # infrastructure. If this task happens to be for a bare slab, then we
        # should take out some extraneous adsorbate information. We should have
        # a separate set of tasks, but that's for future us to fix.
        if self.adsorbate_name == '':
            del self.gasdb_query['fwname.adsorption_site.0']
            del self.gasdb_query['fwname.adsorption_site.1']
            del self.gasdb_query['fwname.adsorption_site.2']
            del self.gasdb_query['fwname.adsorbate_rotation.phi']
            del self.gasdb_query['fwname.adsorbate_rotation.theta']
            del self.gasdb_query['fwname.adsorbate_rotation.psi']
            del self.fw_query['name.adsorption_site.0']
            del self.fw_query['name.adsorption_site.1']
            del self.fw_query['name.adsorption_site.2']
            del self.fw_query['name.adsorbate_rotation.phi']
            del self.fw_query['name.adsorbate_rotation.theta']
            del self.fw_query['name.adsorbate_rotation.psi']

        self.dependency = MakeAdslabFW(adsorption_site=self.adsorption_site,
                                       shift=self.shift,
                                       top=self.top,
                                       dft_settings=self.dft_settings,
                                       adsorbate_name=self.adsorbate_name,
                                       rotation=self.rotation,
                                       mpid=self.mpid,
                                       miller_indices=self.miller_indices,
                                       min_xy=self.min_xy,
                                       slab_generator_settings=self.slab_generator_settings,
                                       get_slab_settings=self.get_slab_settings,
                                       bulk_dft_settings=self.bulk_dft_settings)


class FindSurface(FindCalculation):
    '''
    This task will try to find a surface calculation in either our auxiliary
    Mongo database or our FireWorks database. If the calculation is complete,
    then it will return the results. If the calculation is pending, it will
    wait. If the calculation has not yet been submitted, then it will start the
    calculation.

    Args:
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to get a surface from
        miller_indices          A 3-tuple containing the three Miller indices
                                of the surface you want to find
        shift                   A float indicating the shift of the
                                surface---i.e., the termination that pymatgen
                                finds
        min_height              A float indicating the minimum height of the
                                surface you want to find
        dft_settings            A dictionary containing your DFT settings for
                                the surface relaxation
        bulk_dft_settings       A dictionary containing the DFT settings of
                                the relaxed bulk to enumerate slabs from
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here as a
                                dictionary.
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
    saved output:
        doc     When the calculation is found in our auxiliary Mongo database
                successfully, then this task's output will be the matching
                Mongo document (i.e., dictionary) with various information
                about the system. Some import keys include 'fwid', 'fwname', or
                'results'. This document should  also be able to be turned an
                `ase.Atoms` object using `gaspy.mongo.make_atoms_from_doc`.
    '''
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    min_height = luigi.FloatParameter()
    dft_settings = luigi.DictParameter(SLAB_SETTINGS['vasp'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    # Should explicitly remove the `min_slab_size` key/value from here, because
    # it will be overridden
    slab_generator_settings = deepcopy(SLAB_SETTINGS['slab_generator_settings'])
    del slab_generator_settings['min_slab_size']
    slab_generator_settings = luigi.DictParameter(slab_generator_settings)

    def requires(self):
        '''
        This calculation finder is different because we need to create the
        `ase.Atoms` objects BEFORE performing the queries, because the queries
        will contain the number of atoms. And we don't know the number of atoms
        until we make the `ase.Atoms` object.
        '''
        return FindBulk(mpid=self.mpid, dft_settings=self.bulk_dft_settings)

    def _load_attributes(self):
        '''
        Parses and saves Luigi parameters into various class attributes
        required to run this task, as per the parent class `FindCalculation`
        '''
        # Need to create the slab in order to figure out how many atoms it has,
        # which we'll be using in the queries
        atoms = self._create_surface()
        n_atoms = len(atoms)

        # Required of all `FindCalculation` classes
        self.gasdb_query = {'fwname.calculation_type': 'surface energy optimization',
                            'fwname.mpid': self.mpid,
                            'fwname.miller': self.miller_indices,
                            'fwname.shift': {'$gte': self.shift - 1e-3,
                                             '$lte': self.shift + 1e-3},
                            'atoms.natoms': n_atoms}
        self.fw_query = {'name.calculation_type': 'surface energy optimization',
                         'name.mpid': self.mpid,
                         'name.miller': self.miller_indices,
                         'name.shift': {'$gte': self.shift - 1e-3,
                                        '$lte': self.shift + 1e-3},
                         'name.num_slab_atoms': n_atoms}
        # Parse the DFT settings
        for key, value in self.dft_settings.items():
            # We don't care if these VASP-DFT settings change
            if key != 'isym':
                self.gasdb_query['fwname.dft_settings.%s' % key] = value
                self.fw_query['name.dft_settings.%s' % key] = value

        # We want to pass the atoms object to the `MakeSurfaceFW` task, but
        # Luigi doesn't accept `ase.Atoms` arguments. So we package it into a
        # dictionary/document.
        atoms_doc = make_doc_from_atoms(atoms)
        # Delete some keys that Luigi doesn't like
        del atoms_doc['ctime']
        del atoms_doc['mtime']

        self.dependency = MakeSurfaceFW(atoms_doc=atoms_doc,
                                        mpid=self.mpid,
                                        miller_indices=self.miller_indices,
                                        shift=self.shift,
                                        dft_settings=self.dft_settings)

    def _create_surface(self):
        '''
        This method will create the surface structure to relax

        Returns:
            surface_atoms_constrained   `ase.Atoms` object of the surface to
                                        submit to Fireworks for relaxation
        '''
        # Get the bulk and convert to `pymatgen.Structure` object
        with open(self.input().path, 'rb') as file_handle:
            bulk_doc = pickle.load(file_handle)
        bulk_atoms = make_atoms_from_doc(bulk_doc)
        bulk_structure = AseAtomsAdaptor.get_structure(bulk_atoms)

        # Use pymatgen to turn the bulk into a surface
        sga = SpacegroupAnalyzer(bulk_structure, symprec=0.1)
        bulk_structure = sga.get_conventional_standard_structure()
        gen = SlabGenerator(initial_structure=bulk_structure,
                            miller_index=self.miller_indices,
                            min_slab_size=self.min_height,
                            **self.slab_generator_settings)
        surface_structure = gen.get_slab(self.shift, tol=self.get_slab_settings['tol'])

        # Convert the surface back to an `ase.Atoms` object and constrain
        # subsurface atoms
        surface_atoms = AseAtomsAdaptor.get_atoms(surface_structure)
        surface_atoms_constrained = self.__constrain_surface(surface_atoms)
        return surface_atoms_constrained

    @staticmethod
    def __constrain_surface(atoms, z_cutoff=3.):
        '''
        Constrains the sub-surface atoms of a surface energy slab. This differs
        from `gaspy.atoms_operators.constrain_slab` in that it allows both the
        top and bottom atoms of the slab to be free, not just the top.

        Arg:
            atoms       ASE-atoms class of the slab system
            z_cutoff    The threshold to see if slab atoms are in the same
                        plane as the highest atom in the slab
        Returns:
            atoms   A deep copy of the `atoms` argument, but where the
                    appropriate atoms are constrained
        '''
        # Work on a copy so that we don't modify the original
        atoms = atoms.copy()

        # We'll be making a `mask` list to feed to the `FixAtoms` class. This
        # list should contain a `True` if we want an atom to be constrained,
        # and `False` otherwise
        mask = []

        # Fix any atoms that are `z_cutoff` Angstroms higher than the lowest
        # atom or `z_cutoff` atoms lower than the highest atom.
        z_positions = [atom.position[2] for atom in atoms]
        upper_cutoff = max(z_positions) - z_cutoff
        lower_cutoff = min(z_positions) + z_cutoff
        for atom in atoms:
            if lower_cutoff < atom.position[2] < upper_cutoff:
                mask.append(True)
            else:
                mask.append(False)

        atoms.constraints += [FixAtoms(mask=mask)]
        return atoms
