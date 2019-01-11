'''
This module analyzes our raw data and parses it into various metadata
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
#import numpy as np
#from ase.calculators.singlepoint import SinglePointCalculator
import luigi
from .core import save_task_output, make_task_output_object
from .calculation_finders import FindGas#, FindAdslab
from ..mongo import make_atoms_from_doc
from .. import utils
from .. import defaults

GASDB_PATH = utils.read_rc('gasdb_path')
GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS
SLAB_SETTINGS = defaults.SLAB_SETTINGS
ADSLAB_SETTINGS = defaults.GAS_SETTINGS


#class CalculateAdsorptionEnergy(luigi.Task):
#    '''
#    '''
#    # Adsorbate information
#    adsorbate_name = luigi.Parameter()
#    adsorption_site = luigi.TupleParameter()
#    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
#
#    # Bulk information
#    mpid = luigi.Parameter()
#
#    # Slab information
#    miller_indices = luigi.TupleParameter()
#    shift = luigi.FloatParameter()
#    top = luigi.BoolParameter()
#    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
#    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
#    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
#
#    # VASP settings
#    gas_vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])
#    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
#    adslab_vasp_settings = luigi.DictParameter(ADSLAB_SETTINGS['vasp'])
#
#    def requires(self):
#        return {'CO_gas': FindGas(gas_name='CO', vasp_settings=self.gas_vasp_settings),
#                'H2_gas': FindGas(gas_name='H2', vasp_settings=self.gas_vasp_settings),
#                'H2O_gas': FindGas(gas_name='H2O', vasp_settings=self.gas_vasp_settings),
#                'N2_gas': FindGas(gas_name='N2', vasp_settings=self.gas_vasp_settings),
#                'bare_slab': FindAdslab(adsorption_site=(0., 0., 0.),
#                                        shift=self.shift,
#                                        top=self.top,
#                                        vasp_settings=self.adslab_vasp_settings,
#                                        adsorbate_name='',
#                                        rotation=ADSLAB_SETTINGS['rotation'],
#                                        mpid=self.mpid,
#                                        miller_indices=self.miller_indices,
#                                        min_xy=self.min_xy,
#                                        slab_generator_settings=self.slab_generator_settings,
#                                        get_slab_settings=self.get_slab_settings,
#                                        bulk_vasp_settings=self.bulk_vasp_settings),
#                'adslab': FindAdslab(adsorption_site=self.adsorption_site,
#                                     shift=self.shift,
#                                     top=self.top,
#                                     vasp_settings=self.adslab_vasp_settings,
#                                     adsorbate_name=self.adsorbate_name,
#                                     rotation=self.rotation,
#                                     mpid=self.mpid,
#                                     miller_indices=self.miller_indices,
#                                     min_xy=self.min_xy,
#                                     slab_generator_settings=self.slab_generator_settings,
#                                     get_slab_settings=self.get_slab_settings,
#                                     bulk_vasp_settings=self.bulk_vasp_settings)}
#
#    def run(self):
#        inputs = self.input()
#
#        # Load the gas phase energies
#        gasEnergies = {}
#        gasEnergies['CO'] = make_atoms_from_doc(pickle.load(open(inputs[2].fn, 'rb'))[0]).get_potential_energy()
#        gasEnergies['H2'] = make_atoms_from_doc(pickle.load(open(inputs[3].fn, 'rb'))[0]).get_potential_energy()
#        gasEnergies['H2O'] = make_atoms_from_doc(pickle.load(open(inputs[4].fn, 'rb'))[0]).get_potential_energy()
#        gasEnergies['N2'] = make_atoms_from_doc(pickle.load(open(inputs[5].fn, 'rb'))[0]).get_potential_energy()
#
#        # Load the slab+adsorbate relaxed structures, and take the lowest energy one
#        adslab_docs = pickle.load(open(inputs[0].fn, 'rb'))
#        lowest_energy_adslab = np.argmin([make_atoms_from_doc(doc).get_potential_energy(apply_constraint=False) for doc in adslab_docs])
#        adslab_energy = make_atoms_from_doc(adslab_docs[lowest_energy_adslab]).get_potential_energy(apply_constraint=False)
#
#        # Load the slab relaxed structures, and take the lowest energy one
#        slab_docs = pickle.load(open(inputs[1].fn, 'rb'))
#        lowest_energy_slab = np.argmin([make_atoms_from_doc(doc).get_potential_energy(apply_constraint=False) for doc in slab_docs])
#        slab_energy = np.min([make_atoms_from_doc(doc).get_potential_energy(apply_constraint=False) for doc in slab_docs])
#
#        # Get the per-atom energies as a linear combination of the basis set
#        mono_atom_energies = {'H': gasEnergies['H2']/2.,
#                              'O': gasEnergies['H2O'] - gasEnergies['H2'],
#                              'C': gasEnergies['CO'] - (gasEnergies['H2O']-gasEnergies['H2']),
#                              'N': gasEnergies['N2']/2.}
#
#        # Get the total energy of the stoichiometry amount of gas reference species
#        gas_energy = 0
#        for ads in self.parameters['adsorption']['adsorbates']:
#            gas_energy += np.sum([mono_atom_energies[x] for x in
#                                 utils.ads_dict(ads['name']).get_chemical_symbols()])
#
#        # Calculate the adsorption energy
#        dE = adslab_energy - slab_energy - gas_energy
#
#        # Make an atoms object with a single-point calculator that contains the potential energy
#        adjusted_atoms = make_atoms_from_doc(adslab_docs[lowest_energy_adslab])
#        adjusted_atoms.set_calculator(SinglePointCalculator(adjusted_atoms,
#                                                            forces=adjusted_atoms.get_forces(apply_constraint=False),
#                                                            energy=dE))
#
#        # Write a dictionary with the results and the entries that were used for the calculations
#        # so that fwid/etc for each can be recorded
#        towrite = {'atoms': adjusted_atoms,
#                   'slab+ads': adslab_docs[lowest_energy_adslab],
#                   'slab': slab_docs[lowest_energy_slab],
#                   'gas': {'CO': pickle.load(open(inputs[2].fn, 'rb'))[0],
#                           'H2': pickle.load(open(inputs[3].fn, 'rb'))[0],
#                           'H2O': pickle.load(open(inputs[4].fn, 'rb'))[0],
#                           'N2': pickle.load(open(inputs[5].fn, 'rb'))}}
#
#        # Write the dictionary as a pickle
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump(towrite, open(self.temp_output_path, 'wb'))
#
#        for ads in self.parameters['adsorption']['adsorbates']:
#            print('Finished CalculateEnergy for %s on the %s site of %s %s:  %s eV' % (ads['name'],
#                  self.parameters['adsorption']['adsorbates'][0]['adsorption_site'],
#                  self.parameters['bulk']['mpid'],
#                  self.parameters['slab']['miller'],
#                  dE))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


class CalculateAdsorbateBasisEnergies(luigi.Task):
    '''
    When calculating adsorption energies, we first need the energy of the
    adsorbate. Sometimes the adsorbate does not exist in the gas phase, so we
    can't get the DFT energy. To address this, we can actually calculate the
    adsorbate energy as a sum of basis energies for each atom in the adsorbate.
    For example:  `E(CH3OH)` can be calculated by adding `3*E(C) + 4*E(H) +
    1*E(O)`. To get the energies of the single atoms, we can relax normal gases
    and perform similar algebra, e.g., `E(H) = E(H2)/2` or `E(O) = E(H2O) -
    E(H2)`. This task will calculate the basis energies for H, O, C, and N for
    you so that you can use these energies in other calculations.

    Arg:
        vasp_settings   A dictionary containing the VASP settings you want to
                        use for the DFT relaxations of the gases.
    Returns:
        basis_energies  A dictionary whose keys are the basis elements and
                        whose values are their respective energies, e.g.,
                        {'H': foo, 'O': bar}
    '''
    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def requires(self):
        return {'CO': FindGas(gas_name='CO', vasp_settings=self.vasp_settings),
                'H2': FindGas(gas_name='H2', vasp_settings=self.vasp_settings),
                'H2O': FindGas(gas_name='H2O', vasp_settings=self.vasp_settings),
                'N2': FindGas(gas_name='N2', vasp_settings=self.vasp_settings)}

    def run(self):
        # Load each gas and calculate their energies
        gas_energies = dict.fromkeys(self.input())
        for adsorbate_name, target in self.input().items():
            with open(target.path, 'rb') as file_handle:
                doc = pickle.load(file_handle)
            atoms = make_atoms_from_doc(doc)
            gas_energies[adsorbate_name] = atoms.get_potential_energy()

        # Calculate and save the basis energies from the gas phase energies
        basis_energies = {'H': gas_energies['H2']/2.,
                          'O': gas_energies['H2O'] - gas_energies['H2'],
                          'C': gas_energies['CO'] - (gas_energies['H2O']-gas_energies['H2']),
                          'N': gas_energies['N2']/2.}
        save_task_output(self, basis_energies)

    def output(self):
        return make_task_output_object(self)


#class FingerprintRelaxedAdslab(luigi.Task):
#    '''
#    This class takes relaxed structures from our Pickles, fingerprints them, then adds the
#    fingerprints back to our Pickles
#    '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        Our first requirement is CalculateEnergy, which relaxes the slab+ads system. Our second
#        requirement is to relax the slab+ads system again, but without the adsorbates. We do
#        this to ensure that the "blank slab" we are using in the adsorption calculations has
#        the same number of slab atoms as the slab+ads system.
#        '''
#        # Here, we take the adsorbate off the slab+ads system
#        param = utils.unfreeze_dict(copy.deepcopy(self.parameters))
#        param['adsorption']['adsorbates'] = [OrderedDict(name='',
#                                                         atoms=utils.encode_atoms_to_hex(Atoms('')))]
#        return [CalculateEnergy(self.parameters),
#                SubmitToFW(parameters=param,
#                           calctype='slab+adsorbate')]
#
#    def run(self):
#        ''' We fingerprint the slab+adsorbate system both before and after relaxation. '''
#        # Load the atoms objects for the lowest-energy slab+adsorbate (adslab) system and the
#        # blank slab (slab)
#        calc_e_dict = pickle.load(open(self.input()[0].fn, 'rb'))
#        slab = pickle.load(open(self.input()[1].fn, 'rb'))
#
#        # The atoms object for the adslab prior to relaxation
#        adslab0 = make_atoms_from_doc(calc_e_dict['slab+ads']['initial_configuration'])
#        # The number of atoms in the slab also happens to be the index for the first atom
#        # of the adsorbate (in the adslab system)
#        slab_natoms = slab[0]['atoms']['natoms']
#
#        # If our "adslab" system actually doesn't have an adsorbate, then do not fingerprint
#        if slab_natoms == len(calc_e_dict['atoms']):
#            fp_final = {}
#            fp_init = {}
#        else:
#            # Calculate fingerprints for the initial and final state
#            fp_final = utils.fingerprint_atoms(calc_e_dict['atoms'])
#            fp_init = utils.fingerprint_atoms(adslab0)
#
#        # Save the the fingerprints of the final and initial state as a list in a pickle file
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump([fp_final, fp_init], open(self.temp_output_path, 'wb'))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


#class FingerprintUnrelaxedAdslabs(luigi.Task):
#    '''
#    This class takes unrelaxed slab+adsorbate (adslab) systems from our pickles, fingerprints
#    the adslab, fingerprints the slab (without an adsorbate), and then adds fingerprints back
#    to our Pickles. Note that we fingerprint the slab because we may have had to repeat the
#    original slab to add the adsorbate onto it, and if so then we also need to fingerprint the
#    repeated slab.
#    '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        We call the GenerateAdslabs class twice; once for the adslab, and once for the slab
#        '''
#        # Make a copy of `parameters` for our slab, but then we take off the adsorbate
#        return [GenerateAdSlabs(self.parameters)]
#
#    def run(self):
#        # Load the list of slab+adsorbate (adslab) systems, and the bare slab. Also find the
#        # number of slab atoms
#        adslabs = pickle.load(open(self.input()[0].fn, 'rb'))
#
#        # Fingerprint each adslab
#        for adslab in adslabs:
#            # Don't bother if the adslab happens to be bare
#            if adslab['adsorbate'] == '':
#                fp = {}
#            else:
#                fp = utils.fingerprint_atoms(adslab['atoms'])
#            # Add the fingerprints to the dictionary
#            adslab['fp'] = fp
#
#        # Write
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump(adslabs, open(self.temp_output_path, 'wb'))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))


#class CalculateSlabSurfaceEnergy(luigi.Task):
#    '''
#    This function attempts to calculate the surface energy of a slab using the
#    linear interpolation method. First, we have to find the minimum depth of the slab
#    and then figure out which three slabs we want to ask for. This logic makes the requires
#    function a little more complicated than it otherwise would be
#    '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#
#        # check if bulk exists, and if so pull it.
#        bulk_task = SubmitToFW(calctype='bulk', parameters={'bulk': self.parameters['bulk']})
#        if not(bulk_task.output().exists()):
#            bulk_task.requires()
#            bulk_task.run()
#
#            #If running the SubmitToFW task does not yield an output, then we probably
#            # need to actually require it and let it work through more logic to generate
#            # the necessary bulk structure. We will have to re-run this function after the
#            # the bulk is generated to get the surface energy calculations submitted
#            if not(bulk_task.output().exists):
#                return bulk_task
#
#        # Preparation work with ASE and PyMatGen before we start creating the slabs
#        bulk_doc = pickle.load(open(bulk_task.output().fn, 'rb'))[0]
#
#        # Generate the minimum slab depth slab we can
#        bulk = make_atoms_from_doc(bulk_doc)
#        structure = AseAtomsAdaptor.get_structure(bulk)
#        sga = SpacegroupAnalyzer(structure, symprec=0.1)
#        structure = sga.get_conventional_standard_structure()
#        slab_generate_settings = utils.unfreeze_dict(copy.deepcopy(self.parameters['slab']['slab_generate_settings']))
#        del slab_generate_settings['min_vacuum_size']
#        del slab_generate_settings['min_slab_size']
#        gen = SlabGenerator(structure,
#                            self.parameters['slab']['miller'],
#                            min_vacuum_size=0.,
#                            min_slab_size=0., **slab_generate_settings)
#
#        # Get the number of layers necessary to satisfy the required min_slab_size
#        h = gen._proj_height
#        min_slabs = int(np.ceil(self.parameters['slab']['slab_generate_settings']['min_slab_size']/h))
#
#        # generate the necessary slabs (base thickness + some number of additional layers)
#        req_list = []
#        for nslabs in range(min_slabs, min_slabs+int(self.parameters['slab']['slab_surface_energy_num_layers'])):
#            cur_min_slab_size = h*nslabs
#            gen = SlabGenerator(structure,
#                                self.parameters['slab']['miller'],
#                                min_vacuum_size=self.parameters['slab']['slab_generate_settings']['min_vacuum_size'],
#                                min_slab_size=cur_min_slab_size, **slab_generate_settings)
#            gen.min_slab_size = cur_min_slab_size
#            slab = gen.get_slab(self.parameters['slab']['shift'], tol=self.parameters['slab']['get_slab_settings']['tol'])
#            param_to_submit = utils.unfreeze_dict(copy.deepcopy(dict(self.parameters)))
#            param_to_submit['type'] = 'slab_surface_energy'
#            param_to_submit['slab']['natoms'] = len(slab)
#
#            # Print a warning if the slab is thicker than 80, which means it may not run
#            if len(slab) > self.parameters['bulk']['max_atoms']:
#                print('Surface energy %s %s %s is going to require more than 80 atoms, I hope you know what you are doing!'
#                      % (self.parameters['bulk']['mpid'], self.parameters['slab']['miller'], self.parameters['slab']['shift']))
#                print('aborting!')
#                return
#
#            # Generate the SubmitToFW that will trigger the necessary calculation
#            del param_to_submit['slab']['top']
#            param_to_submit['slab']['slab_generate_settings']['min_slab_size'] = cur_min_slab_size
#            req_list.append(SubmitToFW(calctype='slab_surface_energy', parameters=copy.deepcopy(param_to_submit)))
#
#        # Submit all of the the required slabs
#        return req_list
#
#    def run(self):
#
#        # Load all of the slabs and turn them into atoms objects
#        requirements = self.input()
#
#        doc_list = [pickle.load(open(req.fn, 'rb'))[0] for req in requirements]
#        atoms_list = [make_atoms_from_doc(doc) for doc in doc_list]
#
#        # Pull the number of atoms of each slab
#        number_atoms = [len(atoms) for atoms in atoms_list]
#
#        # Get the energy per cross sectional area for each slab (averaged for top/bottom)
#        energies = [atoms.get_potential_energy() for atoms in atoms_list]
#        energies = energies/np.linalg.norm(np.cross(atoms_list[0].cell[0], atoms_list[0].cell[1]))
#        energies = energies/2
#
#        # Define how to do a linear regression using statsmodel
#        def OLSfit(X, y):
#            data = sm.add_constant(X)
#            mod = sm.OLS(y, data)
#            res = mod.fit()
#            # Return the intercept and the error estimate on the intercept
#            return res.params[0], res.bse[0]
#
#        # Do the linear fit
#        fit = OLSfit(number_atoms, energies)
#
#        # formulate the dictionary and save it as output
#        towrite = copy.deepcopy(doc_list)
#        towrite[0]['processed_data'] = {}
#        towrite[0]['processed_data']['FW_info'] = {}
#        for i in range(len(atoms_list)):
#            towrite[i]['atoms'] = atoms_list[i]
#            towrite[0]['processed_data']['FW_info'][str(len(atoms_list[i]))] = doc_list[i]['fwid']
#        towrite[0]['processed_data']['surface_energy_info'] = {}
#        towrite[0]['processed_data']['surface_energy_info']['intercept'] = fit[0]
#        towrite[0]['processed_data']['surface_energy_info']['intercept_uncertainty'] = fit[1]
#        towrite[0]['processed_data']['surface_energy_info']['num_points'] = len(atoms_list)
#        towrite[0]['processed_data']['surface_energy_info']['energies'] = [atoms.get_potential_energy() for atoms in atoms_list]
#        towrite[0]['processed_data']['surface_energy_info']['num_atoms'] = [len(atoms) for atoms in atoms_list]
#
#        with self.output().temporary_path() as self.temp_output_path:
#            pickle.dump(towrite, open(self.temp_output_path, 'wb'))
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
