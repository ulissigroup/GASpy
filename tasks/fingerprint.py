# Python modules
import sys
import copy
from collections import OrderedDict
import cPickle as pickle
# 3rd party modules
from ase import Atoms
import luigi
# GASpy modules
from generate import AdSlabs
from vasp.mongo import mongo_doc_atoms
import calculate
from submit_to_fw import SubmitToFW
sys.path.append('..')
from gaspy.utils import fingerprint_atoms


LOCAL_DB_PATH = '/global/cscratch1/sd/zulissi/GASpy_DB/'


class RelaxedAdslab(luigi.Task):
    '''
    This class takes relaxed structures from our Pickles, fingerprints them, then adds the
    fingerprints back to our Pickles
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        Our first requirement is calculate.Energy, which relaxes the slab+ads system. Our second
        requirement is to relax the slab+ads system again, but without the adsorbates. We do
        this to ensure that the "blank slab" we are using in the adsorption calculations has
        the same number of slab atoms as the slab+ads system.
        '''
        # Here, we take the adsorbate off the slab+ads system
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='',
                                                         atoms=pickle.dumps(Atoms('')).
                                                         encode('hex'))]
        return [calculate.Energy(self.parameters),
                SubmitToFW(parameters=param,
                           calctype='slab+adsorbate')]

    def run(self):
        ''' We fingerprint the slab+adsorbate system both before and after relaxation. '''
        # Load the atoms objects for the lowest-energy slab+adsorbate (adslab) system and the
        # blank slab (slab)
        adslab = pickle.load(self.input()[0].open())
        slab = pickle.load(self.input()[1].open())

        # The atoms object for the adslab prior to relaxation
        adslab0 = mongo_doc_atoms(adslab['slab+ads']['initial_configuration'])
        # The number of atoms in the slab also happens to be the index for the first atom
        # of the adsorbate (in the adslab system)
        slab_natoms = slab[0]['atoms']['natoms']
        ads_ind = slab_natoms

        # If our "adslab" system actually doesn't have an adsorbate, then do not fingerprint
        if slab_natoms == len(adslab['atoms']):
            fp_final = {}
            fp_init = {}
        else:
            # Calculate fingerprints for the initial and final state
            fp_final = fingerprint_atoms(adslab['atoms'], ads_ind)
            fp_init = fingerprint_atoms(adslab0, ads_ind)

        # Save the the fingerprints of the final and initial state as a list in a pickle file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump([fp_final, fp_init], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))


class UnrelaxedAdslabs(luigi.Task):
    '''
    This class takes unrelaxed slab+adsorbate (adslab) systems from our pickles, fingerprints
    the adslab, fingerprints the slab (without an adsorbate), and then adds fingerprints back
    to our Pickles. Note that we fingerprint the slab because we may have had to repeat the
    original slab to add the adsorbate onto it, and if so then we also need to fingerprint the
    repeated slab.
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We call the GenerateAdslabs class twice; once for the adslab, and once for the slab
        '''
        # Make a copy of `parameters` for our slab, but then we take off the adsorbate
        param_slab = copy.deepcopy(self.parameters)
        param_slab['adsorption']['adsorbates'] = \
                [OrderedDict(name='', atoms=pickle.dumps(Atoms('')).encode('hex'))]
        return [AdSlabs(self.parameters),
                AdSlabs(parameters=param_slab)]

    def run(self):
        # Load the list of slab+adsorbate (adslab) systems, and the bare slab. Also find the
        # number of slab atoms
        adslabs = pickle.load(self.input()[0].open())
        slab = pickle.load(self.input()[1].open())
        expected_slab_atoms = len(slab[0]['atoms'])
        # len(slabs[0]['atoms']['atoms'])*np.prod(eval(adslabs[0]['slabrepeat']))

        # Fingerprint each adslab
        for adslab in adslabs:
            # Don't bother if the adslab happens to be bare
            if adslab['adsorbate'] == '':
                fp = {}
            else:
                fp = fingerprint_atoms(adslab['atoms'], expected_slab_atoms)
            # Add the fingerprints to the dictionary
            for key in fp:
                adslab[key] = fp[key]

        # Write
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adslabs, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
