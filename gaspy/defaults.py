'''
This modules contains default settings for various function and queries used in GASpy and its
submodules.
'''

import copy
from collections import OrderedDict
from ase import Atoms
import ase.constraints
from . import utils


def adsorption_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not take anything out without thinking
    very hard about it.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'mongo_id': '$_id',
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


def adsorption_filters():
    '''
    Not all of our adsorption calculations are "good" ones. Some end up in desorptions,
    dissociations, do not converge, or have ridiculous energies. These are the
    filters we use to sift out these "bad" documents.
    '''
    filters = {}

    # Easy-to-read (and change) filters before we distribute them
    # into harder-to-read (but mongo-readable) structures
    energy_min = -4.            # Minimum adsorption energy [eV]
    energy_max = 4.             # Maximum adsorption energy [eV]
    f_max = 0.5                 # Maximum atomic force [eV/Ang]
    ads_move_max = 1.5          # Maximum distance the adsorbate can move [Ang]
    bare_slab_move_max = 0.5    # Maximum distance that any atom can move on bare slab [Ang]
    slab_move_max = 1.5         # Maximum distance that any slab atom can move after adsorption [Ang]

    # Distribute filters into mongo-readable form
    filters['results.energy'] = {'$gt': energy_min, '$lt': energy_max}
    filters['results.forces'] = {'$not': {'$elemMatch': {'$elemMatch': {'$gt': f_max}}}}
    filters['processed_data.movement_data.max_adsorbate_movement'] = {'$lt': ads_move_max}
    filters['processed_data.movement_data.max_bare_slab_movement'] = {'$lt': bare_slab_move_max}
    filters['processed_data.movement_data.max_surface_movement'] = {'$lt': slab_move_max}
    _calc_settings = calc_settings()
    filters['processed_data.vasp_settings.gga'] = _calc_settings['gga']

    return filters


def catalog_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not take anything out without thinking
    very hard about it.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'shift': '$processed_data.calculation_info.shift',
                    'top': '$processed_data.calculation_info.top',
                    'coordination': '$processed_data.fp_init.coordination',
                    'neighborcoord': '$processed_data.fp_init.neighborcoord',
                    'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination'}
    return fingerprints


def surface_fingerprints():
    '''
    WARNING:  A lot of code depends on this. Do not take anything out without thinking
    very hard about it.

    Returns a dictionary that is meant to be passed to mongo aggregators to create
    new mongo docs. The keys here are the keys for the new mongo doc, and the values
    are where you can find the information from the old mongo docs (in our databases).
    '''
    fingerprints = {'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'intercept': '$processed_data.surface_energy_info.intercept',
                    'intercept_uncertainty': '$processed_data.surface_energy_info.intercept_uncertainty',
                    'initial_configuration': '$initial_configuration',
                    'FW_info': '$processed_data.FW_info'
                    }
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
    filters['results.forces'] = {'$not': {'$elemMatch': {'$elemMatch': {'$gt': f_max}}}}
    filters['processed_data.movement_data.max_surface_movement'] = {'$lt': max_surface_movement}
    _calc_settings = calc_settings()
    filters['processed_data.vasp_settings.gga'] = _calc_settings['gga']

    return filters


def exchange_correlational_settings():
    '''
    Yields a dictionary whose keys are some typical sets of exchange correlationals
    and whose values are dictionaries with the corresponding pseudopotential (pp),
    generalized gradient approximations (ggas), and other pertinent information.

    Credit goes to John Kitchin who wrote vasp.Vasp.xc_defaults, which we copied and put here.
    '''
    xc_settings = {'lda': {'pp': 'LDA'},
                   # GGAs
                   'gga': {'pp': 'GGA'},
                   'pbe': {'pp': 'PBE'},
                   'revpbe': {'pp': 'LDA', 'gga': 'RE'},
                   'rpbe': OrderedDict(gga='RP', pp='PBE'),
                   'am05': {'pp': 'LDA', 'gga': 'AM'},
                   'pbesol': OrderedDict(gga='PS', pp='PBE'),
                   # Meta-GGAs
                   'tpss': {'pp': 'PBE', 'metagga': 'TPSS'},
                   'revtpss': {'pp': 'PBE', 'metagga': 'RTPSS'},
                   'm06l': {'pp': 'PBE', 'metagga': 'M06L'},
                   # vdW-DFs
                   'optpbe-vdw': {'pp': 'LDA', 'gga': 'OR', 'luse_vdw': True,
                                  'aggac': 0.0},
                   'optb88-vdw': {'pp': 'LDA', 'gga': 'BO', 'luse_vdw': True,
                                  'aggac': 0.0, 'param1': 1.1 / 6.0,
                                  'param2': 0.22},
                   'optb86b-vdw': {'pp': 'LDA', 'gga': 'MK', 'luse_vdw': True,
                                   'aggac': 0.0, 'param1': 0.1234,
                                   'param2': 1.0},
                   'vdw-df2': {'pp': 'LDA', 'gga': 'ML', 'luse_vdw': True,
                               'aggac': 0.0, 'zab_vdw': -1.8867},
                   'beef-vdw': {'pp': 'PBE', 'gga': 'BF', 'luse_vdw': True,
                                'zab_vdw': -1.8867, 'lbeefens': True},
                   # hybrids
                   'pbe0': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True},
                   'hse03': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True,
                             'hfscreen': 0.3},
                   'hse06': {'pp': 'LDA', 'gga': 'PE', 'lhfcalc': True,
                             'hfscreen': 0.2},
                   'b3lyp': {'pp': 'LDA', 'gga': 'B3', 'lhfcalc': True,
                             'aexx': 0.2, 'aggax': 0.72,
                             'aggac': 0.81, 'aldac': 0.19},
                   'hf': {'pp': 'PBE', 'lhfcalc': True, 'aexx': 1.0,
                          'aldac': 0.0, 'aggac': 0.0}}
    return xc_settings


def calc_settings(xc='rpbe'):
    '''
    The default calculational settings for GASpy to use.

    Arg:
        xc  A string indicating which exchange correlational to use. This argument
            is used to pick which settings to use within the
            `gaspy.defaults.exchange_correlational_settings()` dictionary, so
            refer to that for valid settings of `xc`.
    Returns:
        settings    An OrderedDict containing the default energy cutoff,
                    VASP pseudo-potential version number (pp_version), and
                    exchange-correlational settings.
    '''
    # Standard settings to use regardless of exchange correlational
    settings = OrderedDict({'encut': 350, 'pp_version': '5.4'})

    # Call on the xc_settings function to define the rest of the settings
    xc_settings = exchange_correlational_settings()
    for key, value in xc_settings[xc].items():
        settings[key] = value

    return settings


def gas_parameters(gasname, settings='rpbe'):
    '''
    Generate some default parameters for a gas and expected relaxation settings

    Args:
        gasname     A string containing the name of the gas
        settings    A string that Vaspy can use to create vasp settings.
                    Or `rpbe` if we want to use that
    '''
    # calc_settings returns a default set of calculational settings, but only if
    # the `settings` argument is a string.
    if isinstance(settings, str):
        settings = calc_settings(settings)

    return OrderedDict(gasname=gasname,
                       relaxed=True,
                       vasp_settings=OrderedDict(ibrion=2,
                                                 nsw=100,
                                                 isif=0,
                                                 kpts=[1, 1, 1],
                                                 ediffg=-0.03,
                                                 **settings))


def bulk_parameters(mpid, settings='rpbe', encut=500., max_atoms=80):
    '''
    Generate some default parameters for a bulk and expected relaxation settings

    Args:
        gasname     A string containing the name of the gas
        settings    A string that Vaspy can use to create vasp settings.
                    Or `rpbe` if we want to use that
        encut       The energy cut-off
    '''
    # calc_settings returns a default set of calculational settings, but only if
    # the `settings` argument is a string.
    if isinstance(settings, str):
        settings = calc_settings(settings)

    # We're getting a handle to a dictionary, so need to copy before modifying
    settings = copy.deepcopy(settings)
    settings['encut'] = encut
    return OrderedDict(mpid=mpid,
                       relaxed=True,
                       max_atoms=max_atoms,
                       vasp_settings=OrderedDict(ibrion=1,
                                                 nsw=100,
                                                 isif=7,
                                                 isym=0,
                                                 ediff=1e-8,
                                                 kpts=[10, 10, 10],
                                                 prec='Accurate',
                                                 **settings))


def slab_parameters(miller, top, shift, settings='rpbe'):
    '''
    Generate some default parameters for a slab and expected relaxation settings

    Args:
        miller      A list of the three miller indices of the slab
        top         A boolean stating whether or not the "top" of the slab is pointing upwards
        shift       As per PyMatGen, the shift is the distance of the planar translation
                    in the z-direction (after the cut). Look up PyMatGen for more details.
        settings    A string that Vaspy can use to create vasp settings.
                    Or `rpbe` if we want to use that
    '''
    # calc_settings returns a default set of calculational settings, but only if
    # the `settings` argument is a string.
    if isinstance(settings, str):
        settings = calc_settings(settings)

    return OrderedDict(miller=miller,
                       top=top,
                       max_miller=2,
                       shift=shift,
                       relaxed=True,
                       vasp_settings=OrderedDict(ibrion=2,
                                                 nsw=100,
                                                 isif=0,
                                                 isym=0,
                                                 kpts=[4, 4, 1],
                                                 lreal='Auto',
                                                 ediffg=-0.03,
                                                 **settings),
                       slab_generate_settings=OrderedDict(min_slab_size=7.,
                                                          min_vacuum_size=20.,
                                                          lll_reduce=False,
                                                          center_slab=True,
                                                          primitive=True,
                                                          max_normal_search=1),
                       get_slab_settings=OrderedDict(tol=0.3,
                                                     bonds=None,
                                                     max_broken_bonds=0,
                                                     symmetrize=False))


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

    ''' Diatomics '''
    # For diatomics (and above), it's a good practice to manually relax the gases
    # and then see how far apart they are. Then put first atom at the origin, and
    # put the second atom directly above it.
    adsorbates['CO'] = Atoms('CO', positions=[[0., 0., 0.],
                                              [0., 0., 1.2]])
    adsorbates['OH'] = Atoms('OH', positions=[[0., 0., 0.],
                                              [0., 0., 0.96]])

    ''' Triatomics '''
    # For OOH, we've found that most of our relaxations resulted in dissociation
    # of at least the hydrogen. As such, we put some hookean springs between
    # the atoms to keep the adsorbate together.
    ooh = Atoms('OOH', positions=[[0., 0., 0.],
                                  [0., 0., 1.55],
                                  [0, 0.94, 1.80]])
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

    # For COH, assumed C binds to surface (index 0), O (index 1), and H(index 2)
    # trying to apply Hookean so that CH bound doesn't dissociate
    cho = Atoms('CHO', positions=[[0., 0., 0.],
                                  [-1.0, 0.2, 0.45],  # position of H
                                  [0.986, 0.6, 0.8]])  # position of O
    cho.set_constraint([ase.constraints.Hookean(a1=0, a2=1, rt=1.59, k=7.),   # Bind CH
                        ase.constraints.Hookean(a1=0, a2=2, rt=1.79, k=5.)])  # Bind CO
    adsorbates['CHO'] = cho

    return adsorbates


def adsorption_parameters(adsorbate,
                          adsorption_site=None,
                          slabrepeat='(1, 1)',
                          num_slab_atoms=0,
                          settings='rpbe'):
    '''
    Generate some default parameters for an adsorption configuration and expected
    relaxation settings

    Args:
        adsorbate       If this is a string, this function will try to find it in the
                        default dictionary of adsorbates. If it is not in the default
                        dictionary, then this function will assume that you have passed
                        it an ase.Atoms object and then act accordingly.
        adsorption_site The cartesian coordinates of the binding atom in the adsorbate
        slabrepeat      The number of times the basic slab has been repeated
        numb_slab_atoms The number of atoms in the slab. We use this number to help
                        differentiate slab and adsorbate atoms (later on).
        settings        A string that Vaspy can use to create vasp settings.
                        Or `rpbe` if we want to use that
    '''
    # calc_settings returns a default set of calculational settings, but only if
    # the `settings` argument is a string.
    if isinstance(settings, str):
        settings = calc_settings(settings)

    # Use EAFP to figure out if the adsorbate that the user passed is in the
    # dictionary of default adsorbates, or if the user supplied an atoms object
    try:
        atoms = adsorbates_dict()[adsorbate]
        name = adsorbate
    except TypeError:
        atoms = adsorbate
        name = adsorbate.get_chemical_formula()

    return OrderedDict(numtosubmit=2,
                       min_xy=4.5,
                       relaxed=True,
                       num_slab_atoms=num_slab_atoms,
                       slabrepeat=slabrepeat,
                       adsorbates=[OrderedDict(name=name,
                                               atoms=utils.encode_atoms_to_hex(atoms),
                                               adsorption_site=adsorption_site)],
                       vasp_settings=OrderedDict(ibrion=2,
                                                 nsw=200,
                                                 isif=0,
                                                 isym=0,
                                                 kpts=[4, 4, 1],
                                                 lreal='Auto',
                                                 ediffg=-0.03,
                                                 symprec=1e-10,
                                                 **settings))
