'''
This submodule contains various tasks that generato ase.atoms.Atoms objects.
The output of all the tasks in this submodule are actually dictionaries (or
"docs" as we define them, which is short for Mongo document). If you want the
atoms object, then use the gaspy.mongo.make_atoms_from_doc function on the
output.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import copy
import math
from math import ceil
from collections import OrderedDict
import pickle
import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.build import rotate
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
import luigi
from .core import make_task_output_object, save_task_output
from .fireworks_submitters import SubmitToFW
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


class GenerateSlabs(luigi.Task):
    '''
    This class uses PyMatGen to create surfaces (i.e., slabs cut from a bulk) from ASE atoms
    objects
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        If the bulk does not need to be relaxed, we simply pull it from Materials Project using
        the `Bulk` class. If it needs to be relaxed, then we submit it to Fireworks.
        '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return GenerateBulk(parameters={'bulk': self.parameters['bulk']})
        elif 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == 'relaxed_bulk':
            return SubmitToFW(calctype='bulk', parameters={'bulk': self.parameters['bulk']})
        else:
            return SubmitToFW(calctype='bulk', parameters={'bulk': self.parameters['bulk']})

    def run(self):
        # Preparation work with ASE and PyMatGen before we start creating the slabs
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            bulk_doc = pickle.load(open(self.input().fn, 'rb'))[0]
        else:
            bulk_docs = pickle.load(open(self.input().fn, 'rb'))
            bulkmin = np.argmin([x['results']['energy'] for x in bulk_docs])
            bulk_doc = bulk_docs[bulkmin]

        # Pull out the fwid of the relaxed bulk (if there is one)
        if not ('unrelaxed' in self.parameters and self.parameters['unrelaxed']):
            bulk_fwid = bulk_doc['fwid']
        else:
            bulk_fwid = None
        bulk = make_atoms_from_doc(bulk_doc)
        structure = AseAtomsAdaptor.get_structure(bulk)
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        structure = sga.get_conventional_standard_structure()
        gen = SlabGenerator(structure,
                            self.parameters['slab']['miller'],
                            **self.parameters['slab']['slab_generate_settings'])
        slabs = gen.get_slabs(**self.parameters['slab']['get_slab_settings'])
        slabsave = []
        for slab in slabs:
            shift = slab.shift

            # Create an atoms class for this particular slab, "atoms_slab"
            atoms_slab = AseAtomsAdaptor.get_atoms(slab)
            # Then reorient the "atoms_slab" class so that the surface of the slab is pointing
            # upwards in the z-direction
            rotate(atoms_slab,
                   atoms_slab.cell[2], (0, 0, 1),
                   atoms_slab.cell[0], [1, 0, 0],
                   rotate_cell=True)
            # Save the slab, but only if it isn't already in the database
            top = True
            tags = {'type': 'slab',
                    'top': top,
                    'mpid': self.parameters['bulk']['mpid'],
                    'miller': self.parameters['slab']['miller'],
                    'shift': shift,
                    'num_slab_atoms': len(atoms_slab),
                    'relaxed': False,
                    'bulk_fwid': bulk_fwid,
                    'slab_generate_settings': self.parameters['slab']['slab_generate_settings'],
                    'get_slab_settings': self.parameters['slab']['get_slab_settings']}
            slabdoc = make_doc_from_atoms(utils.constrain_slab(atoms_slab))
            slabdoc['tags'] = tags
            slabsave.append(slabdoc)

            # If the top of the cut is not identical to the bottom, then save the bottom slab
            # to the database, as well. To do this, we first pull out the sga class of this
            # particular slab, "sga_slab". Again, we use a symmetry finding tolerance of 0.1
            # to be consistent with MP
            sga_slab = SpacegroupAnalyzer(slab, symprec=0.1)
            # Then use the "sga_slab" class to create a list, "symm_ops", that contains classes,
            # which contain matrix and vector operators that may be used to rotate/translate the
            # slab about axes of symmetry
            symm_ops = sga_slab.get_symmetry_operations()
            # Create a boolean, "z_invertible", which will be "True" if the top of the slab is
            # the same as the bottom.
            z_invertible = True in list(map(lambda x: x.as_dict()['matrix'][2][2] == -1, symm_ops))
            # If the bottom is different, then...
            if not z_invertible:
                # flip the slab upside down...
                atoms_slab.wrap()
                atoms_slab.rotate('x', math.pi, rotate_cell=True, center='COM')
                if atoms_slab.cell[2][2] < 0.:
                    atoms_slab.cell[2] = -atoms_slab.cell[2]
                if np.cross(atoms_slab.cell[0], atoms_slab.cell[1])[2] < 0.0:
                    atoms_slab.cell[1] = -atoms_slab.cell[1]
                atoms_slab.wrap()

                # and if it is not in the database, then save it.
                slabdoc = make_doc_from_atoms(utils.constrain_slab(atoms_slab))
                tags = {'type': 'slab',
                        'top': not(top),
                        'mpid': self.parameters['bulk']['mpid'],
                        'miller': self.parameters['slab']['miller'],
                        'shift': shift,
                        'num_slab_atoms': len(atoms_slab),
                        'relaxed': False,
                        'slab_generate_settings': self.parameters['slab']['slab_generate_settings'],
                        'get_slab_settings': self.parameters['slab']['get_slab_settings']}
                slabdoc['tags'] = tags
                slabsave.append(slabdoc)

        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(slabsave, open(self.temp_output_path, 'wb'))

        return

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


class GenerateSiteMarkers(luigi.Task):
    '''
    This class will take a set of slabs, enumerate the adsorption sites on the slab, add a
    marker on the sites (i.e., Uranium), and then save the Uranium+slab systems into our
    pickles
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        If the system we are trying to create markers for is unrelaxed, then we only need
        to create the bulk and surfaces. If the system should be relaxed, then we need to
        submit the bulk and the slab to Fireworks.
        '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return [GenerateSlabs(parameters=OrderedDict(unrelaxed=True,
                                                         bulk=self.parameters['bulk'],
                                                         slab=self.parameters['slab'])),
                    GenerateBulk(parameters={'bulk': self.parameters['bulk']})]
        elif 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == 'relaxed_bulk':
            return [GenerateSlabs(parameters=OrderedDict(unrelaxed=self.parameters['unrelaxed'],
                                                         bulk=self.parameters['bulk'],
                                                         slab=self.parameters['slab'])),
                    SubmitToFW(calctype='bulk',
                               parameters={'bulk': self.parameters['bulk']})]
        else:
            return [SubmitToFW(calctype='slab',
                               parameters=OrderedDict(bulk=self.parameters['bulk'],
                                                      slab=self.parameters['slab'])),
                    SubmitToFW(calctype='bulk',
                               parameters={'bulk': self.parameters['bulk']})]

    def run(self):
        # Defire our marker, a uraniom Atoms object. Then pull out the slabs and bulk
        adsorbate = {'name': 'U', 'atoms': Atoms('U')}
        slab_docs = pickle.load(open(self.input()[0].fn, 'rb'))

        # Initialize `adslabs_to_save`, which will be a list containing marked slabs (i.e.,
        # adslabs) for us to save
        adslabs_to_save = []
        for slab_doc in slab_docs:
            # "slab" [atoms class] is the first slab structure in Aux DB that corresponds
            # to the slab that we are looking at. Note that thise any possible repeats of the
            # slab in the database.
            slab = make_atoms_from_doc(slab_doc)
            # Pull out the fwid of the relaxed slab (if there is one)
            if not ('unrelaxed' in self.parameters and self.parameters['unrelaxed']):
                slab_fwid = slab_doc['fwid']
            else:
                slab_fwid = None

            # Repeat the atoms in the slab to get a cell that is at least as large as the
            # "mix_xy" parameter we set above.
            nx = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[0])))
            ny = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[1])))
            slabrepeat = (nx, ny, 1)
            slab.info['adsorbate_info'] = ''
            slab_repeat = slab.repeat(slabrepeat)

            # Find the adsorption sites. Then for each site we find, we create a dictionary
            # of tags to describe the site. Then we save the tags to our pickles.
            sites = utils.find_adsorption_sites(slab)
            for site in sites:
                # Populate the `tags` dictionary with various information
                if 'unrelaxed' in self.parameters:
                    shift = slab_doc['tags']['shift']
                    top = slab_doc['tags']['top']
                    miller = slab_doc['tags']['miller']
                else:
                    shift = self.parameters['slab']['shift']
                    top = self.parameters['slab']['top']
                    miller = self.parameters['slab']['miller']
                tags = {'type': 'slab+adsorbate',
                        'adsorption_site': str(np.round(site, decimals=2)),
                        'slabrepeat': str(slabrepeat),
                        'adsorbate': adsorbate['name'],
                        'top': top,
                        'miller': miller,
                        'shift': shift,
                        'slab_fwid': slab_fwid,
                        'relaxed': False}
                # Then add the adsorbate marker on top of the slab. Note that we use a local,
                # deep copy of the marker because the marker was created outside of this loop.
                _adsorbate = adsorbate['atoms'].copy()
                # Move the adsorbate onto the adsorption site...
                _adsorbate.translate(site)
                # Put the adsorbate onto the slab and add the adslab system to the tags
                adslab = slab_repeat.copy() + _adsorbate
                tags['atoms'] = adslab

                # Finally, add the information to list of things to save
                adslabs_to_save.append(tags)

        # Save the marked systems to our pickles
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adslabs_to_save, open(self.temp_output_path, 'wb'))

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


class GenerateAdSlabs(luigi.Task):
    '''
    This class takes a set of adsorbate positions from SiteMarkers and replaces
    the marker (a uranium atom) with the correct adsorbate. Adding an adsorbate is done in two
    steps (marker enumeration, then replacement) so that the hard work of enumerating all
    adsorption sites is only done once and reused for every adsorbate
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We need the generated adsorbates with the marker atoms.  We delete
        parameters['adsorption']['adsorbates'] so that every generate_adsorbates_marker call
        looks the same, even with different adsorbates requested in this task
        '''
        parameters_no_adsorbate = utils.unfreeze_dict(copy.deepcopy(self.parameters))
        del parameters_no_adsorbate['adsorption']['adsorbates']
        return GenerateSiteMarkers(parameters_no_adsorbate)

    def run(self):
        # Load the configurations
        adsorbate_configs = pickle.load(open(self.input().fn, 'rb'))

        # For each configuration replace the marker with the adsorbate
        for adsorbate_config in adsorbate_configs:
            # Load the atoms object for the slab and adsorbate
            slab = adsorbate_config['atoms']
            ads = utils.decode_hex_to_atoms(self.parameters['adsorption']['adsorbates'][0]['atoms'])

            # Rotate the adsorbate into place
            #this check is needed if adsorbate = '', in which case there is no adsorbate_rotation
            if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
                ads.euler_rotate(**self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation'])

            # Find the position of the marker/adsorbate and the number of slab atoms
            ads_pos = slab[-1].position
            # Delete the marker on the slab, and then put the slab under the adsorbate.
            # Note that we add the slab to the adsorbate in order to maintain any
            # constraints that may be associated with the adsorbate (because ase only
            # keeps the constraints of the first atoms object).
            del slab[-1]
            ads.translate(ads_pos)

            # If there is a hookean constraining the adsorbate to a local position, we need to adjust
            # it based on ads_pos. We only do this for hookean constraints fixed to a point
            for constraint in ads.constraints:
                dict_repr = constraint.todict()
                if dict_repr['name'] == 'Hookean' and constraint._type == 'point':
                    constraint.origin += ads_pos

            adslab = ads + slab
            adslab.cell = slab.cell
            adslab.pbc = [True, True, True]
            # We set the tags of slab atoms to 0, and set the tags of the adsorbate to 1.
            # In future version of GASpy, we intend to set the tags of co-adsorbates
            # to 2, 3, 4... etc (per co-adsorbate)
            tags = [1]*len(ads)
            tags.extend([0]*len(slab))
            adslab.set_tags(tags)
            # Set constraints for the slab and update the list of dictionaries with
            # the correct atoms object adsorbate name.
            adsorbate_config['atoms'] = utils.constrain_slab(adslab)
            adsorbate_config['adsorbate'] = self.parameters['adsorption']['adsorbates'][0]['name']

            #this check is needed if adsorbate = '', in which case there is no adsorbate_rotation
            if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
                adsorbate_config['adsorbate_rotation'] = self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation']

        # Save the generated list of adsorbate configurations to a pkl file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adsorbate_configs, open(self.temp_output_path, 'wb'))

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
