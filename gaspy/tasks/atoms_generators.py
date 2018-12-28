'''
This submodule contains various tasks that generate `ase.Atoms` objects.
The output of all the tasks in this submodule are actually dictionaries (or
"docs" as we define them, which is short for Mongo document). If you want the
atoms object, then use the gaspy.mongo.make_atoms_from_doc function on the
output.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

#import copy
#from math import ceil
#from collections import OrderedDict
import pickle
#import numpy as np
#from numpy.linalg import norm
#from ase import Atoms
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
import luigi
from .core import save_task_output, make_task_output_object
#from .fireworks_submitters import SubmitToFW
from ..atoms_operators import (make_slabs_from_bulk_atoms,
                               orient_atoms_upwards,
                               constrain_slab,
                               is_structure_invertible,
                               flip_atoms)
from ..mongo import make_doc_from_atoms, make_atoms_from_doc
from .. import utils, defaults

GASDB_PATH = utils.read_rc('gasdb_path')
GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS


class GenerateGas(luigi.Task):
    '''
    Makes a gas-phase atoms object using ASE's g2 collection

    Arg:
        gas_name    A string that can be fed to ase.collection.g2 to create an
                    atoms object (e.g., 'CO', 'OH')
    saved output:
        doc     The atoms object in the format of a dictionary/document
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
    saved output:
        doc     The atoms object in the format of a dictionary/document
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


class GenerateSlabsFromUnrelaxedBulk(luigi.Task):
    '''
    This class enumerates slabs from unrelaxed bulk structures.

    Arg:
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
    '''
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])

    def requires(self):
        return GenerateBulk(mpid=self.mpid)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            bulk_doc = pickle.load(file_handle)
        bulk_atoms = make_atoms_from_doc(bulk_doc)
        slab_structs = make_slabs_from_bulk_atoms(atoms=bulk_atoms,
                                                  miller_indices=self.miller_indices,
                                                  slab_generator_settings=self.slab_generator_settings,
                                                  get_slab_settings=self.get_slab_settings)
        slab_docs = make_slab_docs_from_structs(slab_structs)
        save_task_output(self, slab_docs)

    def output(self):
        return make_task_output_object(self)


def make_slab_docs_from_structs(slab_structures):
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
        docs.append(doc)

        # If slabs are invertible (i.e., are not symmetric about the x-y
        # plane), then flip it and make another document out of it.
        if is_structure_invertible(struct) is True:
            atoms_flipped = flip_atoms(atoms)
            atoms_flipped_constrained = constrain_slab(atoms_flipped)
            doc_flipped = make_doc_from_atoms(atoms_flipped_constrained)
            doc_flipped['shift'] = struct.shift
            doc_flipped['top'] = False
            docs.append(doc_flipped)

    return docs


#class GenerateSlabsFromRelaxedBulk(luigi.Task):
#    '''
#    This class enumerates slabs from relaxed bulk structures.
#
#    Arg:
#        mpid                    A string indicating the Materials Project ID of
#                                the bulk you want to cut a slab from
#        miller_indices          A 3-tuple containing the three Miller indices
#                                of the slabs you want to enumerate
#        bulk_vasp_settings      The vasp settings that should have been used
#                                to relax the bulk structure from which we will
#                                be enumerating slabs.
#        slab_generator_settings We use pymatgen's `SlabGenerator` class to
#                                enumerate surfaces. You can feed the arguments
#                                for that class here as a dictionary.
#        get_slab_settings       We use the `get_slabs` method of pymatgen's
#                                `SlabGenerator` class. You can feed the
#                                arguments for the `get_slabs` method here
#                                as a dictionary.
#    '''
#    mpid = luigi.Parameter()
#    miller_indices = luigi.TupleParameter()
#    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
#    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
#    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
#
#    def requires(self):
#        return GenerateBulk(mpid=self.mpid)
#
#    def run(self):
#        with open(self.input().fn, 'rb') as file_handle:
#            bulk_doc = pickle.load(file_handle)
#        bulk_atoms = make_atoms_from_doc(bulk_doc)
#        slab_structs = make_slabs_from_bulk_atoms(atoms=bulk_atoms,
#                                                  miller_indices=self.miller_indices,
#                                                  slab_generator_settings=self.slab_generator_settings,
#                                                  get_slab_settings=self.get_slab_settings)
#        slab_docs = make_slab_docs_from_structs(slab_structs)
#        save_task_output(self, slab_docs)
#
#    def output(self):
#        return make_task_output_object(self)
#
#class GenerateSiteMarkers(luigi.Task):
#    '''
#    This class will take a set of slabs, enumerate the adsorption sites on the slab, add a
#    marker on the sites (i.e., Uranium), and then save the Uranium+slab systems into our
#    pickles
#    '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        If the system we are trying to create markers for is unrelaxed, then we only need
#        to create the bulk and surfaces. If the system should be relaxed, then we need to
#        submit the bulk and the slab to Fireworks.
#        '''
#        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
#            return [GenerateSlabs(parameters=OrderedDict(unrelaxed=True,
#                                                         bulk=self.parameters['bulk'],
#                                                         slab=self.parameters['slab'])),
#                    GenerateBulk(parameters={'bulk': self.parameters['bulk']})]
#        elif 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == 'relaxed_bulk':
#            return [GenerateSlabs(parameters=OrderedDict(unrelaxed=self.parameters['unrelaxed'],
#                                                         bulk=self.parameters['bulk'],
#                                                         slab=self.parameters['slab'])),
#                    SubmitToFW(calctype='bulk',
#                               parameters={'bulk': self.parameters['bulk']})]
#        else:
#            return [SubmitToFW(calctype='slab',
#                               parameters=OrderedDict(bulk=self.parameters['bulk'],
#                                                      slab=self.parameters['slab'])),
#                    SubmitToFW(calctype='bulk',
#                               parameters={'bulk': self.parameters['bulk']})]
#
#    def run(self):
#        # Defire our marker, a uraniom Atoms object. Then pull out the slabs and bulk
#        adsorbate = {'name': 'U', 'atoms': Atoms('U')}
#        slab_docs = pickle.load(open(self.input()[0].fn, 'rb'))
#
#        # Initialize `adslabs_to_save`, which will be a list containing marked slabs (i.e.,
#        # adslabs) for us to save
#        adslabs_to_save = []
#        for slab_doc in slab_docs:
#            # "slab" [atoms class] is the first slab structure in Aux DB that corresponds
#            # to the slab that we are looking at. Note that thise any possible repeats of the
#            # slab in the database.
#            slab = make_atoms_from_doc(slab_doc)
#            # Pull out the fwid of the relaxed slab (if there is one)
#            if not ('unrelaxed' in self.parameters and self.parameters['unrelaxed']):
#                slab_fwid = slab_doc['fwid']
#            else:
#                slab_fwid = None
#
#            # Repeat the atoms in the slab to get a cell that is at least as large as the
#            # "mix_xy" parameter we set above.
#            nx = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[0])))
#            ny = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[1])))
#            slabrepeat = (nx, ny, 1)
#            slab.info['adsorbate_info'] = ''
#            slab_repeat = slab.repeat(slabrepeat)
#
#            # Find the adsorption sites. Then for each site we find, we create a dictionary
#            # of tags to describe the site. Then we save the tags to our pickles.
#            sites = utils.find_adsorption_sites(slab)
#            for site in sites:
#                # Populate the `tags` dictionary with various information
#                if 'unrelaxed' in self.parameters:
#                    shift = slab_doc['tags']['shift']
#                    top = slab_doc['tags']['top']
#                    miller = slab_doc['tags']['miller']
#                else:
#                    shift = self.parameters['slab']['shift']
#                    top = self.parameters['slab']['top']
#                    miller = self.parameters['slab']['miller']
#                tags = {'type': 'slab+adsorbate',
#                        'adsorption_site': str(np.round(site, decimals=2)),
#                        'slabrepeat': str(slabrepeat),
#                        'adsorbate': adsorbate['name'],
#                        'top': top,
#                        'miller': miller,
#                        'shift': shift,
#                        'slab_fwid': slab_fwid,
#                        'relaxed': False}
#                # Then add the adsorbate marker on top of the slab. Note that we use a local,
#                # deep copy of the marker because the marker was created outside of this loop.
#                _adsorbate = adsorbate['atoms'].copy()
#                # Move the adsorbate onto the adsorption site...
#                _adsorbate.translate(site)
#                # Put the adsorbate onto the slab and add the adslab system to the tags
#                adslab = slab_repeat.copy() + _adsorbate
#                tags['atoms'] = adslab
#
#                # Finally, add the information to list of things to save
#                adslabs_to_save.append(tags)
#
#        # Save the marked systems to our pickles
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump(adslabs_to_save, open(self.temp_output_path, 'wb'))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
#
#
#class GenerateAdSlabs(luigi.Task):
#    '''
#    This class takes a set of adsorbate positions from SiteMarkers and replaces
#    the marker (a uranium atom) with the correct adsorbate. Adding an adsorbate is done in two
#    steps (marker enumeration, then replacement) so that the hard work of enumerating all
#    adsorption sites is only done once and reused for every adsorbate
#    '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        We need the generated adsorbates with the marker atoms.  We delete
#        parameters['adsorption']['adsorbates'] so that every generate_adsorbates_marker call
#        looks the same, even with different adsorbates requested in this task
#        '''
#        parameters_no_adsorbate = utils.unfreeze_dict(copy.deepcopy(self.parameters))
#        del parameters_no_adsorbate['adsorption']['adsorbates']
#        return GenerateSiteMarkers(parameters_no_adsorbate)
#
#    def run(self):
#        # Load the configurations
#        adsorbate_configs = pickle.load(open(self.input().fn, 'rb'))
#
#        # For each configuration replace the marker with the adsorbate
#        for adsorbate_config in adsorbate_configs:
#            # Load the atoms object for the slab and adsorbate
#            slab = adsorbate_config['atoms']
#            ads = utils.decode_hex_to_atoms(self.parameters['adsorption']['adsorbates'][0]['atoms'])
#
#            # Rotate the adsorbate into place
#            #this check is needed if adsorbate = '', in which case there is no adsorbate_rotation
#            if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
#                ads.euler_rotate(**self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation'])
#
#            # Find the position of the marker/adsorbate and the number of slab atoms
#            ads_pos = slab[-1].position
#            # Delete the marker on the slab, and then put the slab under the adsorbate.
#            # Note that we add the slab to the adsorbate in order to maintain any
#            # constraints that may be associated with the adsorbate (because ase only
#            # keeps the constraints of the first atoms object).
#            del slab[-1]
#            ads.translate(ads_pos)
#
#            # If there is a hookean constraining the adsorbate to a local position, we need to adjust
#            # it based on ads_pos. We only do this for hookean constraints fixed to a point
#            for constraint in ads.constraints:
#                dict_repr = constraint.todict()
#                if dict_repr['name'] == 'Hookean' and constraint._type == 'point':
#                    constraint.origin += ads_pos
#
#            adslab = ads + slab
#            adslab.cell = slab.cell
#            adslab.pbc = [True, True, True]
#            # We set the tags of slab atoms to 0, and set the tags of the adsorbate to 1.
#            # In future version of GASpy, we intend to set the tags of co-adsorbates
#            # to 2, 3, 4... etc (per co-adsorbate)
#            tags = [1]*len(ads)
#            tags.extend([0]*len(slab))
#            adslab.set_tags(tags)
#            # Set constraints for the slab and update the list of dictionaries with
#            # the correct atoms object adsorbate name.
#            adsorbate_config['atoms'] = utils.constrain_slab(adslab)
#            adsorbate_config['adsorbate'] = self.parameters['adsorption']['adsorbates'][0]['name']
#
#            #this check is needed if adsorbate = '', in which case there is no adsorbate_rotation
#            if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
#                adsorbate_config['adsorbate_rotation'] = self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation']
#
#        # Save the generated list of adsorbate configurations to a pkl file
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump(adsorbate_configs, open(self.temp_output_path, 'wb'))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
