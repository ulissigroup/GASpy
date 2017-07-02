# Python modules
import sys
import copy
import math
from math import ceil
from collections import OrderedDict
import cPickle as pickle
# 3rd party modules
import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.build import rotate
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.matproj.rest import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from vasp.mongo import mongo_doc, mongo_doc_atoms
import luigi
# GASpy modules
from submit_to_fw import SubmitToFW
sys.path.append('..')
from gaspy.utils import constrain_slab, find_adsorption_sites


LOCAL_DB_PATH = '/global/cscratch1/sd/zulissi/GASpy_DB/'


class Bulk(luigi.Task):
    '''
    This class pulls a bulk structure from Materials Project and then converts it to an ASE
    atoms object
    '''
    parameters = luigi.DictParameter()

    def run(self):
        # Connect to the Materials Project database
        with MPRester("MGOdX3P4nI18eKvE") as m:
            # Pull out the PyMatGen structure and convert it to an ASE atoms object
            structure = m.get_structure_by_material_id(self.parameters['bulk']['mpid'])
            atoms = AseAtomsAdaptor.get_atoms(structure)
            # Dump the atoms object into our pickles
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump([mongo_doc(atoms)], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))


class Gas(luigi.Task):
    parameters = luigi.DictParameter()

    def run(self):
        atoms = g2[self.parameters['gas']['gasname']]
        atoms.positions += 10.
        atoms.cell = [20, 20, 20]
        atoms.pbc = [True, True, True]
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump([mongo_doc(atoms)], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))


class Slabs(luigi.Task):
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
            return Bulk(parameters={'bulk':self.parameters['bulk']})
        else:
            return SubmitToFW(calctype='bulk', parameters={'bulk':self.parameters['bulk']})

    def run(self):
        # Preparation work with ASE and PyMatGen before we start creating the slabs
        atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
        structure = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        structure = sga.get_conventional_standard_structure()
        gen = SlabGenerator(structure,
                            self.parameters['slab']['miller'],
                            **self.parameters['slab']['slab_generate_settings'])
        slabs = gen.get_slabs(**self.parameters['slab']['get_slab_settings'])
        slabsave = []
        for slab in slabs:
            # If this slab is the only one in the set with this miller index, then the shift
            # doesn't matter... so we set the shift as zero.
            if len([a for a in slabs if a.miller_index == slab.miller_index]) == 1:
                shift = 0
            else:
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
            tags = {'type':'slab',
                    'top':top,
                    'mpid':self.parameters['bulk']['mpid'],
                    'miller':self.parameters['slab']['miller'],
                    'shift':shift,
                    'num_slab_atoms':len(atoms_slab),
                    'relaxed':False,
                    'slab_generate_settings':self.parameters['slab']['slab_generate_settings'],
                    'get_slab_settings':self.parameters['slab']['get_slab_settings']}
            slabdoc = mongo_doc(constrain_slab(atoms_slab, len(atoms_slab)))
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
            z_invertible = True in map(lambda x: x.as_dict()['matrix'][2][2] == -1, symm_ops)
            # If the bottom is different, then...
            if not z_invertible:
                # flip the slab upside down...
                atoms_slab.rotate('x', math.pi, rotate_cell=True)

                # and if it is not in the database, then save it.
                slabdoc = mongo_doc(constrain_slab(atoms_slab, len(atoms_slab)))
                tags = {'type':'slab',
                        'top':not(top),
                        'mpid':self.parameters['bulk']['mpid'],
                        'miller':self.parameters['slab']['miller'],
                        'shift':shift,
                        'num_slab_atoms':len(atoms_slab),
                        'relaxed':False,
                        'slab_generate_settings':self.parameters['slab']['slab_generate_settings'],
                        'get_slab_settings':self.parameters['slab']['get_slab_settings']}
                slabdoc['tags'] = tags
                slabsave.append(slabdoc)

        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(slabsave, open(self.temp_output_path, 'w'))

        return

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))


class SiteMarkers(luigi.Task):
    '''
    This class will take a set of slabs, enumerate the adsorption sites on the slab, add a
    marker on the sites (i.e., Uranium), and then save the Uranium+slab systems into our pickles
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        If the system we are trying to create markers for is unrelaxed, then we only need
        to create the bulk and surfaces. If the system should be relaxed, then we need to
        submit the bulk and the slab to Fireworks.
        '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return [Slabs(parameters=OrderedDict(unrelaxed=True,
                                                 bulk=self.parameters['bulk'],
                                                 slab=self.parameters['slab'])),
                    Bulk(parameters={'bulk':self.parameters['bulk']})]
        else:
            return [SubmitToFW(calctype='slab',
                               parameters=OrderedDict(bulk=self.parameters['bulk'],
                                                      slab=self.parameters['slab'])),
                    SubmitToFW(calctype='bulk',
                               parameters={'bulk':self.parameters['bulk']})]

    def run(self):
        # Defire our marker, a uraniom Atoms object. Then pull out the slabs and bulk
        adsorbate = {'name':'U', 'atoms':Atoms('U')}
        slabs = pickle.load(self.input()[0].open())
        bulk = mongo_doc_atoms(pickle.load(self.input()[1].open())[0])

        # Initialize `adslabs_to_save`, which will be a list containing marked slabs (i.e.,
        # adslabs) for us to save
        adslabs_to_save = []
        for slab in slabs:
            # "slab_atoms" [atoms class] is the first slab structure in Aux DB that corresponds
            # to the slab that we are looking at. Note that thise any possible repeats of the
            # slab in the database.
            slab_atoms = mongo_doc_atoms(slab)

            # Repeat the atoms in the slab to get a cell that is at least as large as the
            # "mix_xy" parameter we set above.
            nx = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab_atoms.cell[0])))
            ny = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab_atoms.cell[1])))
            slabrepeat = (nx, ny, 1)
            slab_atoms.info['adsorbate_info'] = ''
            slab_atoms_repeat = slab_atoms.repeat(slabrepeat)

            # Find the adsorption sites. Then for each site we find, we create a dictionary
            # of tags to describe the site. Then we save the tags to our pickles.
            sites = find_adsorption_sites(slab_atoms, bulk)
            for site in sites:
                # Populate the `tags` dictionary with various information
                if 'unrelaxed' in self.parameters:
                    shift = slab['tags']['shift']
                    top = slab['tags']['top']
                    miller = slab['tags']['miller']
                else:
                    shift = self.parameters['slab']['shift']
                    top = self.parameters['slab']['top']
                    miller = self.parameters['slab']['miller']
                tags = {'type':'slab+adsorbate',
                        'adsorption_site':str(np.round(site, decimals=2)),
                        'slabrepeat':str(slabrepeat),
                        'adsorbate':adsorbate['name'],
                        'top':top,
                        'miller':miller,
                        'shift':shift,
                        'relaxed':False}
                # Then add the adsorbate marker on top of the slab. Note that we use a local,
                # deep copy of the marker because the marker was created outside of this loop.
                _adsorbate = adsorbate['atoms'].copy()
                # Move the adsorbate onto the adsorption site...
                _adsorbate.translate(site)
                # Put the adsorbate onto the slab and add the adslab system to the tags
                adslab = slab_atoms_repeat.copy() + _adsorbate
                tags['atoms'] = adslab

                # Finally, add the information to list of things to save
                adslabs_to_save.append(tags)

        # Save the marked systems to our pickles
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adslabs_to_save, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))


class AdSlabs(luigi.Task):
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
        parameters_no_adsorbate = copy.deepcopy(self.parameters)
        del parameters_no_adsorbate['adsorption']['adsorbates']
        return SiteMarkers(parameters_no_adsorbate)

    def run(self):
        # Load the configurations
        adsorbate_configs = pickle.load(self.input().open())

        # For each configuration replace the marker with the adsorbate
        for adsorbate_config in adsorbate_configs:
            # Load the atoms object for the slab and adsorbate
            slab = adsorbate_config['atoms']
            ads = pickle.loads(self.parameters['adsorption']['adsorbates'][0]['atoms'].decode('hex'))
            # Find the position of the marker/adsorbate and the number of slab atoms, which
            # we will use later
            ads_pos = slab[-1].position
            num_slab_atoms = len(slab)
            # Delete the marker on the slab, and then put the adsorbate onto it
            del slab[-1]
            ads.translate(ads_pos)
            adslab = slab + ads
            # Set constraints and update the list of dictionaries with the correct atoms
            # object adsorbate name
            adslab.set_constraint()
            adsorbate_config['atoms'] = constrain_slab(adslab, num_slab_atoms)
            adsorbate_config['adsorbate'] = self.parameters['adsorption']['adsorbates'][0]['name']

        # Save the generated list of adsorbate configurations to a pkl file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adsorbate_configs, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
