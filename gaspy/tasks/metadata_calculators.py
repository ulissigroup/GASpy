'''
This module analyzes our raw data and parses it into various metadata
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import sys
import pickle
import luigi
from .core import save_task_output, make_task_output_object
from .calculation_finders import FindGas, FindAdslab
from ..mongo import make_atoms_from_doc
from .. import utils
from .. import defaults

GASDB_PATH = utils.read_rc('gasdb_path')
GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


class CalculateAdsorptionEnergy(luigi.Task):
    '''
    This task will calculate the adsorption energy of a system you specify.

    Args:
        adsorption_site         A 3-tuple of floats containing the Cartesian
                                coordinates of the adsorption site you want to
                                make a FW for
        shift                   A float indicating the shift of the slab
        top                     A Boolean indicating whether the adsorption
                                site is on the top or the bottom of the slab
        adsorbate_name          A string indicating which adsorbate to use. It
                                should be one of the keys within the
                                `gaspy.defaults.adsorbates` dictionary. If you
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
        gas_vasp_settings       A dictionary containing the VASP settings of
                                the gas relaxation of the adsorbate
        bulk_vasp_settings      A dictionary containing the VASP settings of
                                the relaxed bulk to enumerate slabs from
        adslab_vasp_settings    A dictionary containing your VASP settings
                                for the adslab relaxation
    Returns:
        doc A dictionary with the following keys:
                adsorption_energy   A float indicating the adsorption energy
                fwids               A subdictionary whose keys are 'adslab' and
                                    'slab', and whose values are the FireWork
                                    IDs of the respective calculations.
    '''
    adsorption_site = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    gas_vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])
    bulk_vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])
    adslab_vasp_settings = luigi.DictParameter(ADSLAB_SETTINGS['vasp'])

    def requires(self):
        return {'adsorbate_energy': CalculateAdsorbateEnergy(self.adsorbate_name,
                                                             self.gas_vasp_settings),
                'bare_slab_doc': FindAdslab(adsorption_site=(0., 0., 0.),
                                            shift=self.shift,
                                            top=self.top,
                                            vasp_settings=self.adslab_vasp_settings,
                                            adsorbate_name='',
                                            rotation={'phi': 0., 'theta': 0., 'psi': 0.},
                                            mpid=self.mpid,
                                            miller_indices=self.miller_indices,
                                            min_xy=self.min_xy,
                                            slab_generator_settings=self.slab_generator_settings,
                                            get_slab_settings=self.get_slab_settings,
                                            bulk_vasp_settings=self.bulk_vasp_settings),
                'adslab_doc': FindAdslab(adsorption_site=self.adsorption_site,
                                         shift=self.shift,
                                         top=self.top,
                                         vasp_settings=self.adslab_vasp_settings,
                                         adsorbate_name=self.adsorbate_name,
                                         rotation=self.rotation,
                                         mpid=self.mpid,
                                         miller_indices=self.miller_indices,
                                         min_xy=self.min_xy,
                                         slab_generator_settings=self.slab_generator_settings,
                                         get_slab_settings=self.get_slab_settings,
                                         bulk_vasp_settings=self.bulk_vasp_settings)}

    def run(self):
        with open(self.input()['adsorbate_energy'].path, 'rb') as file_handle:
            ads_energy = pickle.load(file_handle)

        with open(self.input()['bare_slab_doc'].path, 'rb') as file_handle:
            slab_doc = pickle.load(file_handle)
        slab_atoms = make_atoms_from_doc(slab_doc)
        slab_energy = slab_atoms.get_potential_energy(apply_constraint=False)

        with open(self.input()['adslab_doc'].path, 'rb') as file_handle:
            adslab_doc = pickle.load(file_handle)
        adslab_atoms = make_atoms_from_doc(adslab_doc)
        adslab_energy = adslab_atoms.get_potential_energy(apply_constraint=False)

        adsorption_energy = adslab_energy - slab_energy - ads_energy
        doc = {'adsorption_energy': adsorption_energy,
               'fwids': {'adslab': adslab_doc['fwid'],
                         'slab': slab_doc['fwid']}}
        save_task_output(self, doc)

    def output(self):
        return make_task_output_object(self)


class CalculateAdsorbateEnergy(luigi.Task):
    '''
    This task will calculate the energy of an adsorbate via algebraic
    summation/subtraction of a basis set of energies.

    Arg:
        adsorbate_name  A string indicating the name of the adsorbate you
                        want to calculate the energy of
        vasp_settings   A dictionary containing the VASP settings you want to
                        use for the DFT relaxations of the basis set
    Returns:
        energy  The DFT-calculated energy of the adsorbate
    '''
    adsorbate_name = luigi.Parameter()
    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def requires(self):
        return CalculateAdsorbateBasisEnergies(self.vasp_settings)

    def run(self):
        with open(self.input().path, 'rb') as file_handle:
            basis_energies = pickle.load(file_handle)

        # Fetch the adsorbate from our dictionary. If it's not there, yell
        try:
            adsorbate = defaults.adsorbates()[self.adsorbate_name]
        except KeyError as error:
            raise type(error)('You are trying to calculate the adsorbate energy '
                              'of an undefined adsorbate, %s. Please define the '
                              'adsorbate within `gaspy.defaults.adsorbates' %
                              self.adsorbate_name).with_traceback(sys.exc_info()[2])

        energy = sum(basis_energies[atom] for atom in adsorbate.get_chemical_symbols())
        save_task_output(self, energy)

    def output(self):
        return make_task_output_object(self)


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
            gas_energies[adsorbate_name] = atoms.get_potential_energy(apply_constraint=False)

        # Calculate and save the basis energies from the gas phase energies
        basis_energies = {'H': gas_energies['H2']/2.,
                          'O': gas_energies['H2O'] - gas_energies['H2'],
                          'C': gas_energies['CO'] - (gas_energies['H2O']-gas_energies['H2']),
                          'N': gas_energies['N2']/2.}
        save_task_output(self, basis_energies)

    def output(self):
        return make_task_output_object(self)


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
