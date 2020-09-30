'''
This submodule contains the task you can loop through to do a RISM q-series
analysis via SCF calculations instead of full relaxations.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from .core import get_task_output
from .metadata_calculators import CalculateConstantMuAdsorptionEnergy


def submit_rism_scf(adsorbate, catalog_docs,
                    anion_concs, cation_concs,
                    target_fermi=None, **kwargs):
    '''
    Wrapper for submitting RISM-type adsorption calculations given documents
    from the catalog. Specifically, this will submit only SCF calculations.

    Arg:
        adsorbate       A string indicating which adsorbate you want to submit
                        a calculation for. See `gaspy.defaults.adsorbates` for
                        possible values.
        target_fermi    The Fermi energy you want the calculation to be
                        performed at [eV]. If `None`, then we'll do a PZC
                        calculation.
        catalog_docs    Any portion of the list of dictionaries obtained from
                        `gaspy.gasdb.get_catalog_docs` that you want to run.
        anion_concs     A dictionary whose keys are the anions you want in the
                        system and whose values are their concentrations in
                        units of mol/L. What you provide here will override the
                        default in `gaspy.defaults`.
        cation_concs    A dictionary whose keys are the cations you want in the
                        system and whose values are their concentrations in
                        units of mol/L. What you provide here will override the
                        default in `gaspy.defaults`.
        kwargs          If you want to override any arguments for the
                        `gaspy.tasks.metadata_calculators.CalculateConstantMuAdsorptionEnergy`
                        task, then just supply them here. Note that if you
                        supply a value for a field that is inside one of the
                        dictionaries in the `site_docs` argument, the site
                        document will override whatever you provide. This will
                        prevent the user from trying to do calculations on
                        sites that are not real sites, which will probably mess
                        things up downstream.
    '''
    tasks = []
    catalog_docs = copy.deepcopy(catalog_docs)

    # Take out the basic arguments from each site document
    for doc in catalog_docs:
        kwargs['adsorption_site'] = doc['adsorption_site']
        kwargs['mpid'] = doc['mpid']
        kwargs['miller_indices'] = doc['miller']
        kwargs['shift'] = doc['shift']
        kwargs['top'] = doc['top']

        # Define the default RISM settings
        if 'gas_dft_settings' not in kwargs:
            kwargs['gas_dft_settings'] = GAS_SETTINGS['rism']
        if 'bulk_dft_settings' not in kwargs:
            kwargs['bulk_dft_settings'] = BULK_SETTINGS['rism']
        if 'bare_slab_dft_settings' not in kwargs:
            kwargs['bare_slab_dft_settings'] = ADSLAB_SETTINGS['rism']
        if 'adslab_dft_settings' not in kwargs:
            kwargs['adslab_dft_settings'] = ADSLAB_SETTINGS['rism']

        # Make sure we use the same anion and cation concentrations for everything
        for calculation_type in ['gas_dft_settings',
                                 'bare_slab_dft_settings',
                                 'adslab_dft_settings']:
            kwargs[calculation_type]['anion_concs'] = anion_concs
            kwargs[calculation_type]['cation_concs'] = cation_concs

        # If we have a target Fermi level, then make sure we use it for all the
        # calculations
        if target_fermi is not None:
            for calculation_type in ['bare_slab_dft_settings', 'adslab_dft_settings']:
                kwargs[calculation_type]['target_fermi'] = target_fermi
            task = CalculateConstantMuAdsorptionEnergy(adsorbate_name=adsorbate, **kwargs)

        # If we do not have a target Fermi level, then do a PZC
        else:
            task = CalculateChargedRismSCF(adsorbate_name=adsorbate, **kwargs)

        # Submit the jobs to GASpy
        tasks.append(task)
    schedule_tasks(tasks)

    # Parse finished calculations
    finished_docs = []
    lpad = get_launchpad()
    for catalog_doc, task in zip(catalog_docs, tasks):
        try:
            energy_doc = get_task_output(task)
            catalog_doc['adsorption_energy'] = energy_doc['adsorption_energy']
            catalog_doc['anion_concs'] = anion_concs
            catalog_doc['cation_concs'] = cation_concs
            catalog_doc['target_fermi'] = target_fermi
            launch_dirs = {'adslab': lpad.get_launchdir(energy_doc['fwids']['adslab']),
                           'slab': lpad.get_launchdir(energy_doc['fwids']['slab']),
                           'adsorbate': [lpad.get_launchdir(fwid)
                                         for fwid in energy_doc['fwids']['adsorbate']]}
            catalog_doc['launch_dirs'] = launch_dirs
            finished_docs.append(catalog_doc)
        # Print unfinished calculations
        except FileNotFoundError:
            warnings.warn('The following calculation has not finished yet:\n' +
                          '  adsorbate = %s\n' % adsorbate +
                          '  mpid = %s\n' % catalog_doc['mpid'] +
                          '  miller = %s\n' % catalog_doc['miller'] +
                          '  shift = %s\n' % catalog_doc['shift'] +
                          '  top = %s\n' % catalog_doc['top'] +
                          '  site = %s\n' % catalog_doc['adsorption_site'] +
                          '  anions = %s\n' % anion_concs +
                          '  cations = %s\n' % cation_concs +
                          '  fermi = %s' % target_fermi,
                          RuntimeWarning)

    # Print finished calculations
    print('Calculations finished:')
    for doc in finished_docs:
        pprint(doc, indent=1)
    return zip(tasks, finished_docs)


class CalculateChargedRismSCF(CalculateConstantMuAdsorptionEnergy):
    '''
    Does a constant-mu RISM calculation, but only does an SCF instead of a full
    relaxation. Note that this first does a fully relaxed PZC and then an SCF
    using the PZC's relaxed atomic positions.
    '''
    def _rism_requires(self):
        '''
        Typical Luigi tasks have a `requires` method. This task has a series of
        dynamic dependencies. For organizational purposes, we break them up
        into `_*_requires` methods.

        This one returns the RISM calculations.
        '''
        reqs = {}

        # For some reason, Luigi might not run the `_seed_calc_requires` before
        # running this method. If this happens, then call it manually.
        try:
            pzc_task = self.requirements['pzc']
        except KeyError:
            pzc_task = self._seed_calc_requires()['pzc']

        # Fetch the PZC results we need to seed the RISM calculations with
        pzc_doc = get_task_output(pzc_task)
        fwids = pzc_doc['fwids']
        slab_doc, slab_starting_fermi = self.__parse_pzc_info(fwids['slab'])
        adslab_doc, adslab_starting_fermi = self.__parse_pzc_info(fwids['adslab'])

        # Get and feed the results of the PZC relaxation for the slab
        pruned_slab_doc = self.prune_atoms_doc(slab_doc)
        bare_slab_dft_settings = unfreeze_dict(copy.deepcopy(self.bare_slab_dft_settings))
        bare_slab_starting_charge = self.__calculate_tot_charge(self.adslab_dft_settings['target_fermi'],
                                                                slab_starting_fermi,
                                                                pruned_slab_doc)
        bare_slab_dft_settings['tot_charge'] = bare_slab_starting_charge
        bare_slab_dft_settings['calcmode'] = 'scf'
        reqs['bare_slab_doc'] = FindRismAdslab(atoms_dict=pruned_slab_doc,
                                               adsorption_site=(0., 0., 0.),
                                               shift=self.shift,
                                               top=self.top,
                                               dft_settings=bare_slab_dft_settings,
                                               adsorbate_name='',
                                               rotation={'phi': 0., 'theta': 0., 'psi': 0.},
                                               mpid=self.mpid,
                                               miller_indices=self.miller_indices,
                                               min_xy=self.min_xy,
                                               slab_generator_settings=self.slab_generator_settings,
                                               get_slab_settings=self.get_slab_settings,
                                               bulk_dft_settings=self.bulk_dft_settings,
                                               max_fizzles=self.max_fizzles)

        # Get and feed the results of the PZC relaxation for the adslab
        pruned_adslab_doc = self.prune_atoms_doc(adslab_doc)
        adslab_dft_settings = unfreeze_dict(copy.deepcopy(self.adslab_dft_settings))
        adslab_starting_charge = self.__calculate_tot_charge(self.adslab_dft_settings['target_fermi'],
                                                             adslab_starting_fermi,
                                                             pruned_adslab_doc)
        adslab_dft_settings['tot_charge'] = adslab_starting_charge
        adslab_dft_settings['calcmode'] = 'scf'
        reqs['adslab_doc'] = FindRismAdslab(atoms_dict=pruned_adslab_doc,
                                            adsorption_site=self.adsorption_site,
                                            shift=self.shift,
                                            top=self.top,
                                            dft_settings=adslab_dft_settings,
                                            adsorbate_name=self.adsorbate_name,
                                            rotation=self.rotation,
                                            mpid=self.mpid,
                                            miller_indices=self.miller_indices,
                                            min_xy=self.min_xy,
                                            slab_generator_settings=self.slab_generator_settings,
                                            get_slab_settings=self.get_slab_settings,
                                            bulk_dft_settings=self.bulk_dft_settings,
                                            max_fizzles=self.max_fizzles)

        # Get the adsorbate energy
        reqs['adsorbate_energy'] = CalculateAdsorbateEnergy(self.adsorbate_name,
                                                            self.gas_dft_settings,
                                                            max_fizzles=self.max_fizzles)

        self._save_requirements(reqs)
        return reqs
