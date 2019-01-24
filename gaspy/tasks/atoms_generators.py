'''
This submodule contains various tasks that generate `ase.Atoms` objects.
The output of all the tasks in this submodule are actually dictionaries (or
"docs" as we define them, which is short for Mongo document). If you want the
atoms object, then use the gaspy.mongo.make_atoms_from_doc function on the
output.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
import luigi
import ase
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from .core import save_task_output, make_task_output_object
from ..mongo import make_doc_from_atoms, make_atoms_from_doc
from ..atoms_operators import (make_slabs_from_bulk_atoms,
                               orient_atoms_upwards,
                               constrain_slab,
                               is_structure_invertible,
                               flip_atoms,
                               tile_atoms,
                               find_adsorption_sites,
                               add_adsorbate_onto_slab)
from .. import utils, defaults

GASDB_PATH = utils.read_rc('gasdb_path')
GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()
ADSORBATES = defaults.adsorbates()


class GenerateGas(luigi.Task):
    '''
    Makes a gas-phase atoms object using ASE's g2 collection

    Arg:
        gas_name    A string that can be fed to ase.collection.g2 to create an
                    atoms object (e.g., 'CO', 'OH')
    Returns:
        doc     The atoms object in the format of a dictionary/document. This
                document can be turned into an `ase.Atoms` object with the
                `gaspy.mongo.make_atoms_from_doc` function.
    '''
    gas_name = luigi.Parameter()

    def run(self):
        atoms = g2[self.gas_name]
        atoms.positions += 10.
        atoms.cell = [20, 20, 20]
        atoms.pbc = [True, True, True]

        doc = make_doc_from_atoms(atoms)
        save_task_output(self, doc)

    def output(self):
        return make_task_output_object(self)


class GenerateBulk(luigi.Task):
    '''
    This class pulls a bulk structure from Materials Project and then converts
    it to an ASE atoms object

    Arg:
        mpid    A string indicating what the Materials Project ID (mpid) to
                base this bulk on
    Returns:
        doc     The atoms object in the format of a dictionary/document. This
                document can be turned into an `ase.Atoms` object with the
                `gaspy.mongo.make_atoms_from_doc` function.
    '''
    mpid = luigi.Parameter()

    def run(self):
        with MPRester(utils.read_rc('matproj_api_key')) as rester:
            structure = rester.get_structure_by_material_id(self.mpid)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        doc = make_doc_from_atoms(atoms)
        save_task_output(self, doc)

    def output(self):
        return make_task_output_object(self)


class GenerateSlabs(luigi.Task):
    '''
    This class enumerates slabs from relaxed bulk structures.

    Args:
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to cut a slab from
        miller_indices          A 3-tuple containing the three Miller indices
                                of the slabs you want to enumerate
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here
                                as a dictionary.
        bulk_vasp_settings      A dictionary containing the VASP settings of
                                the relaxed bulk to enumerate slabs from
    Returns:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about slabs. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These documents also contain
                the following fields:
                    fwids   A subdictionary containing the FWIDs of the
                            prerequisite calculations
                    shift   Float indicating the shift/termination of the slab
                    top     Boolean indicating whether or not the slab is
                            oriented upwards with respect to the way it was
                            enumerated originally by pymatgen
    '''
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        from .calculation_finders import FindBulk   # local import to avoid import errors
        return FindBulk(mpid=self.mpid)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            bulk_doc = pickle.load(file_handle)
        bulk_atoms = make_atoms_from_doc(bulk_doc)
        slab_structs = make_slabs_from_bulk_atoms(atoms=bulk_atoms,
                                                  miller_indices=self.miller_indices,
                                                  slab_generator_settings=self.slab_generator_settings,
                                                  get_slab_settings=self.get_slab_settings)
        slab_docs = self._make_slab_docs_from_structs(slab_structs, bulk_doc['fwid'])
        save_task_output(self, slab_docs)

    @staticmethod
    def _make_slab_docs_from_structs(slab_structures, fwid):
        '''
        This function will take a list of pymatgen.Structure slabs, convert them
        into `ase.Atoms` objects, orient the slabs upwards, fix the subsurface
        atoms, and then turn those atoms objects into dictionaries (i.e.,
        documents). This function will also enumerate and return new documents for
        invertible slabs that you give it, so the number of documents you get out
        may be greater than the number of structures you put in.

        Arg:
            slab_structures     A list of pymatgen.Structure objects. They should
                                probably be created by the
                                `make_slabs_from_bulk_atoms` function, but you do
                                you.
            fwid                An integer for the FireWorks ID of the calculation
                                used to relax the bulk from which we are
                                enumerating the slab.
        Returns:
            docs    A list of dictionaries (also known as "documents", because
                    they'll eventually be put into Mongo as documents) that contain
                    information about slabs. These documents can be fed to the
                    `gaspy.mongo.make_atoms_from_docs` function to be turned
                    into `ase.Atoms` objects. These documents also contain
                    the 'shift' and 'top' fields to indicate the shift/termination
                    of the slab and whether or not the slab is oriented upwards
                    with respect to the way it was enumerated originally by
                    pymatgen.
        '''
        docs = []
        for struct in slab_structures:
            atoms = AseAtomsAdaptor.get_atoms(struct)
            atoms = orient_atoms_upwards(atoms)

            # Convert each slab into dictionaries/documents
            atoms_constrained = constrain_slab(atoms)
            doc = make_doc_from_atoms(atoms_constrained)
            doc['shift'] = struct.shift
            doc['top'] = True
            doc['fwids'] = {'bulk': fwid}
            docs.append(doc)

            # If slabs are invertible (i.e., are not symmetric about the x-y
            # plane), then flip it and make another document out of it.
            if is_structure_invertible(struct) is True:
                atoms_flipped = flip_atoms(atoms)
                atoms_flipped_constrained = constrain_slab(atoms_flipped)
                doc_flipped = make_doc_from_atoms(atoms_flipped_constrained)
                doc_flipped['shift'] = struct.shift
                doc_flipped['top'] = False
                doc_flipped['fwids'] = {'bulk': fwid}
                docs.append(doc_flipped)

        return docs

    def output(self):
        return make_task_output_object(self)


class GenerateAdsorptionSites(luigi.Task):
    '''
    This task will enumerate all of the adsorption sites from the slabs that
    match the given MPID, miller indices, and slab enumeration settings. It
    will then place a Uranium atom at each of the sites so we can visualize it.

    Args:
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
    Returns:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about the sites. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These objects have a uranium atom
                placed at the adsorption site, and the uranium is tagged with
                a `1`. These documents also contain the following fields:
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
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateSlabs(mpid=self.mpid,
                             miller_indices=self.miller_indices,
                             slab_generator_settings=self.slab_generator_settings,
                             get_slab_settings=self.get_slab_settings,
                             bulk_vasp_settings=self.bulk_vasp_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            slab_docs = pickle.load(file_handle)

        # For each slab, tile it and then find all the adsorption sites
        docs_sites = []
        for slab_doc in slab_docs:
            slab_atoms = make_atoms_from_doc(slab_doc)
            slab_atoms_tiled, slab_repeat = tile_atoms(atoms=slab_atoms,
                                                       min_x=self.min_xy,
                                                       min_y=self.min_xy)
            sites = find_adsorption_sites(slab_atoms_tiled)

            # Place a uranium atom on the adsorption site and then tag it with
            # a `1`, which is our way of saying that it is an adsorbate
            for site in sites:
                adsorbate = ase.Atoms('U')
                adsorbate.translate(site)
                adslab_atoms = slab_atoms_tiled.copy() + adsorbate
                adslab_atoms[-1].tag = 1

                # Turn the atoms into a document, then save it
                doc = make_doc_from_atoms(adslab_atoms)
                doc['fwids'] = slab_doc['fwids']
                doc['shift'] = slab_doc['shift']
                doc['top'] = slab_doc['top']
                doc['slab_repeat'] = slab_repeat
                doc['adsorption_site'] = site
                docs_sites.append(doc)
        save_task_output(self, docs_sites)

    def output(self):
        return make_task_output_object(self)


class GenerateAdslabs(luigi.Task):
    '''
    This class takes a set of adsorbate positions from the
    `GenerateAdsorptionSites` task and replaces the marker (a uranium atom)
    with the correct adsorbate.

    Args:
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
    Returns:
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
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateAdsorptionSites(mpid=self.mpid,
                                       miller_indices=self.miller_indices,
                                       min_xy=self.min_xy,
                                       slab_generator_settings=self.slab_generator_settings,
                                       get_slab_settings=self.get_slab_settings,
                                       bulk_vasp_settings=self.bulk_vasp_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            site_docs = pickle.load(file_handle)

        # Get and rotate the adsorbate
        adsorbate = ADSORBATES[self.adsorbate_name].copy()
        adsorbate.euler_rotate(**self.rotation)

        # Fetch each slab and then replace the Uranium marker with the
        # adsorbate
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


class GenerateAllSitesFromBulk(luigi.Task):
    '''
    This task will enumerate a set of adsorption sites, and then it will find
    these sites in the `catalog` Mongo collection. If any site is not in the
    catalog, then this task will insert it to the catalog.

    Luigi normally likes unit tasks (i.e., a task that adds only one site to
    the catalog, instead of a bunch), but this would end up spawing millions
    of tasks, which Luigi is not fast at handling. So we instead make this
    task, so Luigi only has to handle O(X0,000) tasks.

    Args:
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to enumerate sites from
        max_miller              An integer indicating the maximum Miller index
                                to be enumerated
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
    Returns:
        all_site_docs   A concatenated list of all of the Mongo documents
                        (i.e., dictionaries) generated by multiple calls
                        to the `GenerateAdsorptionSites` task. Refer to that
                        task for specifics regarding the structure of the
                        dictionaries. This task will also add the 'miller'
                        key to each dictionary, and the values will be a
                        tuple of the Miller indices
    '''
    mpid = luigi.Parameter()
    max_miller = luigi.IntParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return _EnumerateDistinctFacets(mpid=self.mpid,
                                        max_miller=self.max_miller,
                                        bulk_vasp_settings=self.bulk_vasp_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            distinct_millers = pickle.load(file_handle)

        # Enumerate the adsorption sites on all the distinct facets
        site_generators = []
        for miller in distinct_millers:
            site_generator = GenerateAdsorptionSites(mpid=self.mpid,
                                                     miller_indices=miller,
                                                     min_xy=self.min_xy,
                                                     slab_generator_settings=self.slab_generator_settings,
                                                     get_slab_settings=self.get_slab_settings,
                                                     bulk_vasp_settings=self.bulk_vasp_settings)
            site_generators.append(site_generator)
        # Yield all the dynamic dependencies at once so that Luigi will run
        # them in parallel instead of sequentially
        outputs_of_generators = yield site_generators

        # Concatenate, append, and save the sites we just enumerated
        all_site_docs = []
        for generator, generator_output in zip(site_generators, outputs_of_generators):
            with open(generator_output.path, 'rb') as file_handle:
                site_docs = pickle.load(file_handle)
            for doc in site_docs:
                doc['miller'] = generator.miller_indices
            all_site_docs.extend(site_docs)
        save_task_output(self, all_site_docs)

    def output(self):
        return make_task_output_object(self)


class _EnumerateDistinctFacets(luigi.Task):
    '''
    This task will enumerate the symmetrically distinct facets of a bulk
    material.

    Args:
        mpid                A string indicating the Materials Project ID, e.g.,
                            'mp-2'
        max_miller          An integer indicating the maximum Miller index to
                            be enumerated
        bulk_vasp_settings  A dictionary containing the VASP settings of the
                            relaxed bulk to enumerate slabs from
    Returns:
        distinct_millers    A list of the distinct Miller indices, where the
                            Miller indices are 3-long sequences of integers.
    '''
    mpid = luigi.Parameter()
    max_miller = luigi.IntParameter()
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        from .calculation_finders import FindBulk   # local import to avoid import errors
        return FindBulk(mpid=self.mpid, vasp_settings=self.bulk_vasp_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            bulk_doc = pickle.load(file_handle)

        # Convert the bulk into a `pytmatgen.Structure` and then standardize it
        # for consistency
        bulk_atoms = make_atoms_from_doc(bulk_doc)
        bulk_structure = AseAtomsAdaptor.get_structure(bulk_atoms)
        sga = SpacegroupAnalyzer(bulk_structure, symprec=0.1)
        bulk_struct_standard = sga.get_conventional_standard_structure()

        # Enumerate and save the distinct Miller indices
        distinct_millers = get_symmetrically_distinct_miller_indices(bulk_struct_standard,
                                                                     self.max_miller)
        save_task_output(self, distinct_millers)

    def output(self):
        return make_task_output_object(self)
