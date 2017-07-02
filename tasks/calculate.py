# Python modules
import sys
import copy
from collections import OrderedDict
import cPickle as pickle
# 3rd party modules
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import luigi
from vasp.mongo import mongo_doc_atoms
# GASpy modules
from submit_to_fw import SubmitToFW
sys.path.append('..')
from gaspy.utils import ads_dict


LOCAL_DB_PATH = '/global/cscratch1/sd/zulissi/GASpy_DB/'


class Energy(luigi.Task):
    '''
    This class attempts to return the adsorption energy of a configuration relative to
    stoichiometric amounts of CO, H2, H2O
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We need the relaxed slab, the relaxed slab+adsorbate, and relaxed CO/H2/H2O gas
        structures/energies
        '''
        # Initialize the list of things that need to be done before we can calculate the
        # adsorption enegies
        toreturn = []

        # First, we need to relax the slab+adsorbate system
        toreturn.append(SubmitToFW(parameters=self.parameters, calctype='slab+adsorbate'))

        # Then, we need to relax the slab. We do this by taking the adsorbate off and
        # replacing it with '', i.e., nothing. It's still labeled as a 'slab+adsorbate'
        # calculation because of our code infrastructure.
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='', atoms=pickle.dumps(Atoms('')).encode('hex'))]
        toreturn.append(SubmitToFW(parameters=param, calctype='slab+adsorbate'))

        # Lastly, we need to relax the base gases.
        for gasname in ['CO', 'H2', 'H2O']:
            param = copy.deepcopy({'gas':self.parameters['gas']})
            param['gas']['gasname'] = gasname
            toreturn.append(SubmitToFW(parameters=param, calctype='gas'))

        # Now we put it all together.
        #print('Checking for/submitting relaxations for %s %s' % (self.parameters['bulk']['mpid'], self.parameters['slab']['miller']))
        return toreturn

    def run(self):
        inputs = self.input()

        # Load the gas phase energies
        gasEnergies = {}
        gasEnergies['CO'] = mongo_doc_atoms(pickle.load(inputs[2].open())[0]).get_potential_energy()
        gasEnergies['H2'] = mongo_doc_atoms(pickle.load(inputs[3].open())[0]).get_potential_energy()
        gasEnergies['H2O'] = mongo_doc_atoms(pickle.load(inputs[4].open())[0]).get_potential_energy()

        # Load the slab+adsorbate relaxed structures, and take the lowest energy one
        slab_ads = pickle.load(inputs[0].open())
        lowest_energy_slab = np.argmin(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_ads))
        slab_ads_energy = mongo_doc_atoms(slab_ads[lowest_energy_slab]).get_potential_energy()

        # Load the slab relaxed structures, and take the lowest energy one
        slab_blank = pickle.load(inputs[1].open())
        lowest_energy_blank = np.argmin(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_blank))
        slab_blank_energy = np.min(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_blank))

        # Get the per-atom energies as a linear combination of the basis set
        mono_atom_energies = {'H':gasEnergies['H2']/2.,
                              'O':gasEnergies['H2O']-gasEnergies['H2'],
                              'C':gasEnergies['CO']-(gasEnergies['H2O']-gasEnergies['H2'])}

        # Get the total energy of the stoichiometry amount of gas reference species
        gas_energy = 0
        for ads in self.parameters['adsorption']['adsorbates']:
            gas_energy += np.sum(map(lambda x: mono_atom_energies[x],
                                     ads_dict(ads['name']).get_chemical_symbols()))

        # Calculate the adsorption energy
        dE = slab_ads_energy - slab_blank_energy - gas_energy

        # Make an atoms object with a single-point calculator that contains the potential energy
        adjusted_atoms = mongo_doc_atoms(slab_ads[lowest_energy_slab])
        adjusted_atoms.set_calculator(SinglePointCalculator(adjusted_atoms,
                                                            forces=adjusted_atoms.get_forces(),
                                                            energy=dE))

        # Write a dictionary with the results and the entries that were used for the calculations
        # so that fwid/etc for each can be recorded
        towrite = {'atoms':adjusted_atoms,
                   'slab+ads':slab_ads[lowest_energy_slab],
                   'slab':slab_blank[lowest_energy_blank],
                   'gas':{'CO':pickle.load(inputs[2].open())[0],
                          'H2':pickle.load(inputs[3].open())[0],
                          'H2O':pickle.load(inputs[4].open())[0]}}

        # Write the dictionary as a pickle
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(towrite, open(self.temp_output_path, 'w'))

        for ads in self.parameters['adsorption']['adsorbates']:
            print('Finished CalculateEnergy for %s on the %s site of %s %s:  %s eV' \
                  % (ads['name'],
                     self.parameters['adsorption']['adsorbates'][0]['adsorption_site'],
                     self.parameters['bulk']['mpid'],
                     self.parameters['slab']['miller'],
                     dE))

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
