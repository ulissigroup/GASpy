'''
This submodule contains various Luigi tasks that are meant to be used to make
FireWorks rockets for adsorptions---i.e., start DFT relaxations
of slab+adsorbate systems.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from collections import OrderedDict
import random
import luigi
import numpy as np
from ... import defaults
from ...gasdb import get_unsimulated_catalog_docs
from ..core import FingerprintRelaxedAdslab, SubmitToFW

DEFAULT_MAX_ROCKETS = 20


class AllSitesOnSurfaces(luigi.WrapperTask):
    '''
    This task will find all of the unique sites on a set of surfaces
    and then create FireWorks rockets out of them, i.e., submit
    for DFT calculations.

    Luigi args:
        ads_list                    A list of lists, where the lists contain strings of the
                                    adsorbates you want to simulate.
        adsorption_rotation_list    A list of dictionaries, each describing a rotation. Acceptable
                                    are [] which will default to no rotation, or a list of euler
                                    rotations such as [{'phi': 0.0, 'theta': 0.0, 'psi': 0.0}]
        mpid_list                   A list of strings indicating the mpid numbers you want to
                                    simulate
        miller_list                 A list of lists of integers indicating the miller indices you
                                    want to simulate.
        xc                          A string indicating the cross-correlational you want to use.
        encut                       A float indicating the energy cutoff (eV) you want to be used
                                    for the slab and adslab relaxations.
        bulk_encut                  A float indicating the energy cutoff (eV) you want to be used
                                    for the bulk relaxation.
        max_bulk_atoms              A positive integer indicating the maximum number of atoms you
                                    want present in the bulk relaxation.
        max_rockets                 A positive integer indicating the maximum number of sites you
                                    want to submit to FireWorks. If the number of possible
                                    site/adsorbate combinations is greater than the maximum number
                                    of submissions, then submission priority is assigned randomly.
    '''
    adsorbates_list = luigi.ListParameter()
    adsorbate_rotation_list = luigi.ListParameter()
    mpid_list = luigi.ListParameter()
    miller_list = luigi.ListParameter()
    xc = luigi.Parameter(defaults.XC)
    encut = luigi.FloatParameter(defaults.ADSLAB_ENCUT)
    bulk_encut = luigi.FloatParameter(defaults.BULK_ENCUT)
    slab_encut = luigi.FloatParameter(defaults.SLAB_ENCUT)
    pp_version = luigi.Parameter(defaults.PP_VERSION)
    max_bulk_atoms = luigi.IntParameter(defaults.MAX_NUM_BULK_ATOMS)
    max_rockets = luigi.IntParameter(DEFAULT_MAX_ROCKETS)

    def requires(self):
        '''
        Get all of the sites in the catalog that match the mpid/miller combination
        specified in the arguments, and then queue a `FingerprintRelaxedAdslab`, which
        should eventually build/queue a rocket build for the adslab.
        '''

        # Turn the mpids and millers into sets because Luigi doesn't have set parameters.
        # Note that we standardize the miller indices to deal with variable syntax.
        mpids_set = set(self.mpid_list)
        millers_set = set(_standardize_miller(miller) for miller in self.miller_list)

        # Iterate through a different set of documents for each adsorbate,
        # because the unsimulated catalog of sites is different for each adsorbate
        parameters_list = []
        for adsorbates in self.adsorbates_list:
            for doc in get_unsimulated_catalog_docs(adsorbates, self.adsorbate_rotation_list):

                # Create the simulation parameters from the site if the site falls
                # within the set of mpid and millers we are looking for.
                if doc['mpid'] in mpids_set and _standardize_miller(doc['miller']) in millers_set:
                    parameters = _make_adslab_parameters_from_doc(doc, adsorbates,
                                                                  encut=self.encut,
                                                                  bulk_encut=self.bulk_encut,
                                                                  slab_encut=self.slab_encut,
                                                                  xc=self.xc,
                                                                  pp_version=self.pp_version,
                                                                  max_bulk_atoms=self.max_bulk_atoms)
                    parameters_list.append(parameters)

        tasks = _make_relaxation_tasks_from_parameters(parameters_list, max_rockets=self.max_rockets)
        return tasks


def _standardize_miller(miller):
    '''
    This function will take either a list or string of Miller indices
    and turn them into a string with no spaces. We do this to make
    sure that Miller indices that we compare are standardize and
    can therefore be queried against each other consistently.

    Arg:
        miller  Either a string, list, or tuple of Miller index integers,
                e.g., '[1, 1, 1]', '[1,1, 1]', [1, 1, 1], (1, 1, 1), or '(1, 1,1)'
    Returns:
        standard_miller A string-formatted version of the Miller indices
                        with no spaces, e.g., '[1,1,1]'
    '''
    # Turn the argument into a string
    if isinstance(miller, list) or isinstance(miller, tuple):
        miller = str(list(miller))
    elif isinstance(miller, str):
        pass
    else:
        raise TypeError('The miller index you provided is not a list, tuple, or string.')

    # Turn parentheses into brackets
    characters = []
    for character in list(miller):
        if character == '(':
            character = '['
        elif character == ')':
            character = ']'
        characters.append(character)

    # Get rid of spaces
    standard_miller = ''.join(character for character in characters if character != ' ')

    return standard_miller


def _make_adslab_parameters_from_doc(doc, adsorbates,
                                     encut=defaults.ADSLAB_ENCUT,
                                     bulk_encut=defaults.BULK_ENCUT,
                                     slab_encut=defaults.SLAB_ENCUT,
                                     xc=defaults.XC,
                                     pp_version=defaults.PP_VERSION,
                                     max_bulk_atoms=defaults.MAX_NUM_BULK_ATOMS):
    '''
    This function creates the `parameters` dictionary that many gaspy
    tasks need to create/submit FireWorks rockets.

    It creates the parameters in a way that we try to submit only one
    calculation per document argument.

    Args:
        doc             A dictionary with the following keys:  'mpid',
                        'miller', 'top', 'shift', and 'adsorption_site'.
                        Should probably come from `gaspy.gasdb.get_catalog_docs`
                        or something like that.
        adsorbates      A list of strings indicating which adsorbates you
                        want to simulate the adsorption of.
        encut           The energy cutoff you want to use for the adslab
                        relaxation.
        slab_encut      The energy cutoff of the slab calculation that you
                        plan to use as the basis of your adslab calculation.
        bulk_encut      The energy cutoff of the bulk calculation that you
                        plan to use as the basis of your slab.
        xc              The exchange correlational you want to use.
        pp_version      A string indicating which version of VASP you want to use.
        max_bulk_atoms  The maximum number of atoms in the corresponding bulk
                        relaxation of system you want to make a
                        rocket/calculation for.
    '''
    # Sometimes we don't care about rotation. If that happens, then just use
    # the default rotation.
    try:
        adsorbate_rotation = doc['adsorbate_rotation']
    except KeyError:
        adsorbate_rotation = defaults.ROTATION

    parameters = OrderedDict.fromkeys(['bulk', 'slab', 'adsorption', 'gas'])
    parameters['bulk'] = defaults.bulk_parameters(mpid=doc['mpid'],
                                                  settings=xc,
                                                  encut=bulk_encut,
                                                  pp_version=pp_version,
                                                  max_atoms=max_bulk_atoms)
    parameters['slab'] = defaults.slab_parameters(miller=doc['miller'],
                                                  top=doc['top'],
                                                  shift=doc['shift'],
                                                  settings=xc,
                                                  encut=slab_encut,
                                                  pp_version=pp_version)
    parameters['adsorption'] = defaults.adsorption_parameters(adsorbate=adsorbates[0],
                                                              adsorption_site=doc['adsorption_site'],
                                                              adsorbate_rotation=adsorbate_rotation,
                                                              settings=xc,
                                                              encut=encut,
                                                              pp_version=pp_version)
    parameters['gas'] = defaults.gas_parameters(adsorbates[0],
                                                settings=xc,
                                                encut=encut,
                                                pp_version=pp_version)
    return parameters


def _make_relaxation_tasks_from_parameters(parameters_list, max_rockets=DEFAULT_MAX_ROCKETS):
    '''
    This function will turn a list of parameters into a generator that
    yields a `FingerprintRelaxedAdslab` task for each set of parameters.
    We execute the fingerprinting task because that requires a relaxation,
    and thus the creation/submission of a FireWorks rocket.

    Args:
        parameters_list A list of OrderedDictionaries whose
                        keys are 'bulk', 'slab', 'adsorption', and 'gas'.
                        The values should be the parameters returned
                        by `gaspy.defaults.*_parameters`.
        max_rockets     An integer indicating the maximum number of
                        rockets that you want to submit at a time.
    Returns:
        tasks   A list of `FingerprintRelaxedAdslab` instances,
                where each element corresponds to each set of
                parameters you've passed as arguments.
    '''
    # Pick the parameters/sites that we'll be using to create rockets.
    # EAFP in case we have less rockets to make than `max_rockets`
    try:
        parameters_list = random.sample(parameters_list, max_rockets)
    except ValueError:
        random.shuffle(parameters_list)

    # Create the rockets
    tasks = []
    for parameters in parameters_list:
        task = FingerprintRelaxedAdslab(parameters=parameters)
        tasks.append(task)
    return tasks


class AllSitesOnSurfacesBlaster(luigi.WrapperTask):
    '''
    This task will find all of the unique sites on a set of surfaces
    and then create FireWorks rockets out of them, i.e., submit
    for DFT calculations.

    It creates the parameters in such a way that we submit every single site
    that's possible, but we also have no idea how many sites that will be.

    Luigi args:
        ads_list        A list of lists, where the lists contain strings of the adsorbates
                        you want to simulate.
        adsorption_rotation_list
                        A list of dictionaries, each describing a rotation. Acceptable are
                        [] which will default to no rotation, or a list of euler rotations
                        such as [{'phi':0.0,'theta':0.0,'psi':0.0}]
        mpid_list       A list of strings indicating the mpid numbers you want to simulate
        miller_list     A list of lists of integers indicating the miller indices you want
                        to simulate.
        xc              A string indicating the cross-correlational you want to use.
        encut           A float indicating the energy cutoff (eV) you want to be used for
                        the slab and adslab relaxations.
        bulk_encut      A float indicating the energy cutoff (eV) you want to be used for
                        the bulk relaxation.
        max_bulk_atoms  A positive integer indicating the maximum number of atoms you want
                        present in the bulk relaxation.
        max_rockets     A positive integer indicating the maximum number of sites you
                        want to submit to FireWorks. If the number of possible
                        site/adsorbate combinations is greater than the maximum number
                        of submissions, then submission priority is assigned randomly.
    '''
    adsorbates_list = luigi.ListParameter()
    adsorbate_rotation_list = luigi.ListParameter()
    mpid_list = luigi.ListParameter()
    miller_list = luigi.ListParameter()
    xc = luigi.Parameter(defaults.XC)
    encut = luigi.FloatParameter(defaults.ADSLAB_ENCUT)
    bulk_encut = luigi.FloatParameter(defaults.BULK_ENCUT)
    slab_encut = luigi.FloatParameter(defaults.SLAB_ENCUT)
    pp_version = luigi.Parameter(defaults.PP_VERSION)
    max_bulk_atoms = luigi.IntParameter(defaults.MAX_NUM_BULK_ATOMS)
    max_rockets = luigi.IntParameter(DEFAULT_MAX_ROCKETS)

    def requires(self):
        '''
        Get all of the sites in the catalog that match the mpid/miller combination
        specified in the arguments, and then queue a `FingerprintRelaxedAdslab`, which
        should eventually build/queue a rocket build for the adslab.
        '''
        # Turn the mpids and millers into sets because Luigi doesn't have set parameters.
        # Note that we standardize the miller indices to deal with variable syntax.
        mpids_set = set(self.mpid_list)
        millers_set = set(self.miller_list)

        # Iterate through a different set of documents for each adsorbate,
        # because the unsimulated catalog of sites is different for each adsorbate
        parameters_list = []
        for adsorbates in self.adsorbates_list:
            # Create the simulation parameters from the site if the site falls
            # within the set of mpid and millers we are looking for.
            for mpid in mpids_set:
                for miller in millers_set:
                    for adsorbate_rotation in self.adsorbate_rotation_list:
                        parameters = _make_adslab_parameters_for_blasting(mpid,
                                                                          miller,
                                                                          adsorbate_rotation,
                                                                          adsorbates,
                                                                          encut=self.encut,
                                                                          bulk_encut=self.bulk_encut,
                                                                          slab_encut=self.slab_encut,
                                                                          xc=self.xc,
                                                                          pp_version=self.pp_version,
                                                                          max_bulk_atoms=self.max_bulk_atoms)
                        parameters_list.append(parameters)

        tasks = [SubmitToFW(calctype='slab+adsorbate', parameters=parameters)
                 for parameters in parameters_list]
        return tasks


def _make_adslab_parameters_for_blasting(mpid, miller, adsorbate_rotation, adsorbates,
                                         encut=defaults.ADSLAB_ENCUT,
                                         bulk_encut=defaults.BULK_ENCUT,
                                         slab_encut=defaults.SLAB_ENCUT,
                                         xc=defaults.XC,
                                         pp_version=defaults.PP_VERSION,
                                         max_bulk_atoms=defaults.MAX_NUM_BULK_ATOMS):
    '''
    This helper function makes the `parameters` dictionary that many gaspy
    tasks need to create/submit FireWorks rockets.

    It creates the parameters in such a way that we submit every single site
    that's possible, but we also have no idea how many sites that will be.

    Args:
        doc             A dictionary with the following keys:  'mpid',
                        'miller', 'top', 'shift', and 'adsorption_site'.
                        Should probably come from `gaspy.gasdb.get_catalog_docs`
                        or something like that.
        adsorbates      A list of strings indicating which adsorbates you
                        want to simulate the adsorption of.
        encut           The energy cutoff you want to use for the adslab
                        relaxation.
        slab_encut      The energy cutoff of the slab calculation that you
                        plan to use as the basis of your adslab calculation.
        bulk_encut      The energy cutoff of the bulk calculation that you
                        plan to use as the basis of your slab.
        xc              The exchange correlational you want to use.
        pp_version      A string indicating which version of VASP you want to use.
        max_bulk_atoms  The maximum number of atoms in the corresponding bulk
                        relaxation of system you want to make a
                        rocket/calculation for.
    '''
    parameters = OrderedDict.fromkeys(['bulk', 'slab', 'adsorption', 'gas'])
    parameters['bulk'] = defaults.bulk_parameters(mpid=mpid,
                                                  settings=xc,
                                                  encut=bulk_encut,
                                                  pp_version=pp_version,
                                                  max_atoms=max_bulk_atoms)
    parameters['slab'] = defaults.slab_parameters(miller=miller,
                                                  top=True,
                                                  shift=np.nan,
                                                  settings=xc,
                                                  encut=slab_encut,
                                                  pp_version=pp_version)
    parameters['adsorption'] = defaults.adsorption_parameters(adsorbate=adsorbates[0],
                                                              adsorption_site=[],
                                                              adsorbate_rotation=adsorbate_rotation,
                                                              settings=xc,
                                                              encut=encut,
                                                              pp_version=pp_version)
    parameters['gas'] = defaults.gas_parameters(adsorbates[0],
                                                settings=xc,
                                                encut=encut,
                                                pp_version=pp_version)

    # This means that every site/shift for a miller index / structure will match
    parameters['adsorption']['adsorbates'][0]['fp'] = {}

    # submit as many sites as we find (including multiple locations, shifts, and top/bottoms)
    parameters['adsorption']['numtosubmit'] = 100
    return parameters
