'''
This modules contains default settings for various function and queries used in GASpy and its
submodules.
'''

import warnings
from collections import OrderedDict
from ase import Atoms
import ase.constraints


# Vasp pseudopotential version
PP_VERSION = '5.4'

# A dictionary whose keys are some typical sets of exchange correlationals and
# whose values are dictionaries with the corresponding pseudopotential (pp),
# generalized gradient approximations (ggas), and other pertinent information.
# Credit goes to John Kitchin who wrote vasp.Vasp.xc_defaults, which we copied
# and put here.
XC_SETTINGS = OrderedDict(lda=OrderedDict(pp='LDA'),
                          gga=OrderedDict(pp='GGA'),    # GGAs
                          pbe=OrderedDict(pp='PBE'),
                          revpbe=OrderedDict(pp='LDA', gga='RE'),
                          rpbe=OrderedDict(gga='RP', pp='PBE'),
                          am05=OrderedDict(pp='LDA', gga='AM'),
                          pbesol=OrderedDict(gga='PS', pp='PBE'),
                          # Meta-GGAs
                          tpss=OrderedDict(pp='PBE', metagga='TPSS'),
                          revtpss=OrderedDict(pp='PBE', metagga='RTPSS'),
                          m06l=OrderedDict(pp='PBE', metagga='M06L'),
                          # vdW-DFs
                          optpbe_vdw=OrderedDict(pp='LDA', gga='OR',
                                                 luse_vdw=True, aggac=0.0),
                          optb88_vdw=OrderedDict(pp='LDA', gga='BO',
                                                 luse_vdw=True, aggac=0.0,
                                                 param1=1.1 / 6.0, param2=0.22),
                          optb86b_vdw=OrderedDict(pp='LDA', gga='MK',
                                                  luse_vdw=True, aggac=0.0,
                                                  param1=0.1234, param2=1.0),
                          vdw_df2=OrderedDict(pp='LDA', gga='ML', luse_vdw=True,
                                              aggac=0.0, zab_vdw=-1.8867),
                          beef_vdw=OrderedDict(pp='PBE', gga='BF', luse_vdw=True,
                                               zab_vdw=-1.8867, lbeefens=True),
                          # hybrids
                          pbe0=OrderedDict(pp='LDA', gga='PE', lhfcalc=True),
                          hse03=OrderedDict(pp='LDA', gga='PE', lhfcalc=True,
                                            hfscreen=0.3),
                          hse06=OrderedDict(pp='LDA', gga='PE', lhfcalc=True,
                                            hfscreen=0.2),
                          b3lyp=OrderedDict(pp='LDA', gga='B3', lhfcalc=True,
                                            aexx=0.2, aggax=0.72, aggac=0.81, aldac=0.19),
                          hf=OrderedDict(pp='PBE', lhfcalc=True, aexx=1.0,
                                         aldac=0.0, aggac=0.0))

# Our default exchange correlational
XC = 'rpbe'

# The default settings we use to do DFT calculations of gases
GAS_SETTINGS = OrderedDict(vasp=OrderedDict(ibrion=2,
                                            nsw=100,
                                            isif=0,
                                            kpts=(1, 1, 1),
                                            ediffg=-0.03,
                                            encut=350.,
                                            pp_version=PP_VERSION,
                                            **XC_SETTINGS[XC]))

# The default settings we use to do DFT calculations of bulks
BULK_SETTINGS = OrderedDict(max_atoms=80,
                            vasp=OrderedDict(ibrion=1,
                                             nsw=100,
                                             isif=7,
                                             isym=0,
                                             ediff=1e-8,
                                             kpts=(10, 10, 10),
                                             prec='Accurate',
                                             encut=500.,
                                             pp_version=PP_VERSION,
                                             **XC_SETTINGS[XC]))

# The default settings we use to enumerate slabs, along with the subsequent DFT
# settings. The 'slab_generator_settings' are passed to the `SlabGenerator`
# class in pymatgen, and the `get_slab_settings` are passed to the `get_slab`
# method of that class.
SLAB_SETTINGS = OrderedDict(max_miller=2,
                            vasp=OrderedDict(ibrion=2,
                                             nsw=100,
                                             isif=0,
                                             isym=0,
                                             kpts=[4, 4, 1],
                                             lreal='Auto',
                                             ediffg=-0.03,
                                             encut=350.,
                                             pp_version=PP_VERSION,
                                             **XC_SETTINGS[XC]),
                            slab_generator_settings=OrderedDict(min_slab_size=7.,
                                                                min_vacuum_size=20.,
                                                                lll_reduce=False,
                                                                center_slab=True,
                                                                primitive=True,
                                                                max_normal_search=1),
                            get_slab_settings=OrderedDict(tol=0.3,
                                                          bonds=None,
                                                          max_broken_bonds=0,
                                                          symmetrize=False))

# The default settings we use to enumerate adslab structures, along with the
# subsequent DFT settings. `mix_xy` is the minimum with of the slab (Angstroms)
# before we enumerate adsorption sites on it.
ADSLAB_SETTINGS = OrderedDict(min_xy=4.5,
                              vasp=OrderedDict(ibrion=2,
                                               nsw=200,
                                               isif=0,
                                               isym=0,
                                               kpts=[4, 4, 1],
                                               lreal='Auto',
                                               ediffg=-0.03,
                                               symprec=1e-10,
                                               encut=350.,
                                               pp_version=PP_VERSION,
                                               **XC_SETTINGS[XC]))

# We use surrogate models to make predictions of DFT information. This is the
# tag associated with our default model.
MODEL = 'model0'


def adsorbates_dict():
    '''
    This function is intended to be used to generate (and store) a library of adsorbates
    in dictionary form, where the key is a string for the adsorbate and the value is an
    ase Atoms object (with constraints, where applicable).

    When making new entries for this dictionary, we recommend "pointing"
    the adsorbate upwards in the z-direction.
    '''
    ''' Nullatomic '''
    adsorbates = {}
    adsorbates[''] = Atoms()

    ''' Monatomics '''
    # Uranium is a place-holder for an adsorbate
    adsorbates['U'] = Atoms('U')
    # Put the hydrogen half an angstrom below the origin to help in adsorb onto the surface
    adsorbates['H'] = Atoms('H', positions=[[0., 0., -0.5]])
    adsorbates['O'] = Atoms('O')
    adsorbates['C'] = Atoms('C')
    adsorbates['N'] = Atoms('N')

    ''' Diatomics '''
    # For diatomics (and above), it's a good practice to manually relax the gases
    # and then see how far apart they are. Then put first atom at the origin, and
    # put the second atom directly above it.
    adsorbates['CO'] = Atoms('CO', positions=[[0., 0., 0.],
                                              [0., 0., 1.2]])
    adsorbates['OH'] = Atoms('OH', positions=[[0., 0., 0.],
                                              [0.92, 0., 0.32]])

    ''' Triatomics '''
    # For OOH, we've found that most of our relaxations resulted in dissociation
    # of at least the hydrogen. As such, we put some hookean springs between
    # the atoms to keep the adsorbate together.
    ooh = Atoms('OOH', positions=[[0., 0., 0.],
                                  [1.28, 0., 0.67],
                                  [1.44, -0.96, 0.81]])
    ooh.set_constraint([ase.constraints.Hookean(a1=0, a2=1, rt=1.6, k=10.),   # Bind OO
                        ase.constraints.Hookean(a1=1, a2=2, rt=1.37, k=5.)])  # Bind OH
    adsorbates['OOH'] = ooh
    # For CHO, assumed C binds to surface (index 0), O (index 1), and H(index 2).
    # Trying to apply Hookean so that CH bound doesn't dissociate. Actual structure is H-C-O
    cho = Atoms('CHO', positions=[[0., 0., 1.],
                                  [-0.94, 0.2, 1.7],  # position of H
                                  [0.986, 0.6, 1.8]])  # position of O
    cho.set_constraint([ase.constraints.Hookean(a1=0, a2=1, rt=1.59, k=5.),
                        # Bind CH, initially used k=7, lowered to 5
                        ase.constraints.Hookean(a1=0, a2=2, rt=1.79, k=5.)])  # Bind CO
    adsorbates['CHO'] = cho

    return adsorbates


def adsorption_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not add or remove anything out without
    thinking very hard about it. If you do add something, consider changing
    the ignore_keys in `gaspy.gasdb.get_unsimulated_catalog_docs`.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'_id': 0,
                    'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'shift': '$processed_data.calculation_info.shift',
                    'top': '$processed_data.calculation_info.top',
                    'coordination': '$processed_data.fp_final.coordination',
                    'neighborcoord': '$processed_data.fp_final.neighborcoord',
                    'nextnearestcoordination': '$processed_data.fp_final.nextnearestcoordination',
                    'energy': '$results.energy',
                    'adsorbates': '$processed_data.calculation_info.adsorbate_names',
                    'adslab_calculation_date': '$processed_data.FW_info.adslab_calculation_date'}
    return fingerprints


def adsorption_filters(adsorbates):
    '''
    Not all of our adsorption calculations are "good" ones. Some end up in desorptions,
    dissociations, do not converge, or have ridiculous energies. These are the
    filters we use to sift out these "bad" documents.

    Arg:
        adsorbates  A list of the adsorbates that you need to
                    be present in each document's corresponding atomic
                    structure. Note that if you pass a list with two adsorbates,
                    then you will only get matches for structures with *both*
                    of those adsorbates; you will *not* get structures
                    with only one of the adsorbates.
    '''
    filters = {}

    # Easy-to-read (and change) filters before we distribute them
    # into harder-to-read (but mongo-readable) structures
    f_max = 0.5                 # Maximum atomic force [eV/Ang]
    ads_move_max = 1.5          # Maximum distance the adsorbate can move [Ang]
    bare_slab_move_max = 0.5    # Maximum distance that any atom can move on bare slab [Ang]
    slab_move_max = 1.5         # Maximum distance that any slab atom can move after adsorption [Ang]
    if adsorbates == ['CO']:
        energy_min = -7.
        energy_max = 5.
    elif adsorbates == ['H']:
        energy_min = -5.
        energy_max = 5.
    elif adsorbates == ['O']:
        energy_min = -4.
        energy_max = 9.
    elif adsorbates == ['OH']:
        energy_min = -3.5
        energy_max = 4.
    elif adsorbates == ['OOH']:
        energy_min = 0.
        energy_max = 9.
    else:
        energy_min = -50.
        energy_max = 50.
        warnings.warn('You are using adsorption document filters for a set of adsorbates that '
                      'we have not yet established valid energy bounds for, yet. We are accepting '
                      'anything in the range between %i and %i eV.' % (energy_min, energy_max))

    # Distribute filters into mongo-readable form
    filters['results.energy'] = {'$gt': energy_min, '$lt': energy_max}
    filters['results.fmax'] = {'$lt': f_max}
    filters['processed_data.movement_data.max_adsorbate_movement'] = {'$lt': ads_move_max}
    filters['processed_data.movement_data.max_bare_slab_movement'] = {'$lt': bare_slab_move_max}
    filters['processed_data.movement_data.max_surface_movement'] = {'$lt': slab_move_max}
    filters['processed_data.vasp_settings.gga'] = XC_SETTINGS[XC]['gga']

    return filters


def catalog_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not add or remove anything out without
    thinking very hard about it. If you do add something, consider changing
    the ignore_keys in `gaspy.gasdb.get_unsimulated_catalog_docs`.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'_id': 0,
                    'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'shift': '$processed_data.calculation_info.shift',
                    'top': '$processed_data.calculation_info.top',
                    'natoms': '$atoms.natoms',
                    'coordination': '$processed_data.fp_init.coordination',
                    'neighborcoord': '$processed_data.fp_init.neighborcoord',
                    'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination',
                    'adsorption_site': '$processed_data.calculation_info.adsorption_site'}
    return fingerprints


def surface_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not take anything out without thinking
    very hard about it.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'_id': 0,
                    'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'intercept': '$processed_data.surface_energy_info.intercept',
                    'intercept_uncertainty': '$processed_data.surface_energy_info.intercept_uncertainty',
                    'initial_configuration': '$initial_configuration',
                    'FW_info': '$processed_data.FW_info'}
    return fingerprints


def surface_filters():
    '''
    Not all of our surface calculations are "good" ones. Some do not converge
    or have end up having a lot of movement. These are the filters we use to sift
    out these "bad" documents.
    '''
    filters = {}

    # Easy-to-read (and change) filters before we distribute them
    # into harder-to-read (but mongo-readable) structures
    f_max = 0.5                 # Maximum atomic force [eV/Ang]
    max_surface_movement = 0.5  # Maximum distance that any atom can move [Ang]

    # Distribute filters into mongo-readable form
    filters['results.fmax'] = {'$lt': f_max}
    filters['processed_data.movement_data.max_surface_movement'] = {'$lt': max_surface_movement}
    filters['processed_data.vasp_settings.gga'] = XC_SETTINGS[XC]['gga']

    return filters
