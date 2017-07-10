import copy
from collections import OrderedDict
import cPickle as pickle
from ase import Atoms
from vasp import Vasp


def xc_settings(xc):
    '''
    This function is where we populate the default calculation settings we want for each
    specific xc (exchange correlational)
    '''
    if xc == 'rpbe':
        settings = OrderedDict(gga='RP', pp='PBE')
    else:
        settings = OrderedDict(Vasp.xc_defaults[xc])

    return settings


def slab_parameters(miller, top, shift, settings='beef-vdw'):
    ''' Generate some default parameters for a slab and expected relaxation settings '''
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


def gas_parameters(gasname, settings='beef-vdw'):
    ''' Generate some default parameters for a gas and expected relaxation settings '''
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


def calc_settings(xc):
    '''
    This function defines the default calculational settings for GASpy to use
    '''
    # Standard settings to use regardless of xc (exchange correlational)
    settings = OrderedDict({'encut': 350, 'pp_version': '5.4'})

    # Call on the xc_settings function to define the rest of the settings
    default_settings = xc_settings(xc)
    for key in default_settings:
        settings[key] = default_settings[key]

    return settings


def bulk_parameters(mpid, settings='beef-vdw', encutBulk=500.):
    ''' Generate some default parameters for a bulk and expected relaxation settings '''
    if isinstance(settings, str):
        settings = calc_settings(settings)
    # We're getting a handle to a dictionary, so need to copy before modifying
    settings = copy.deepcopy(settings)
    settings['encut'] = encutBulk
    return OrderedDict(mpid=mpid,
                       relaxed=True,
                       max_atoms=50,
                       vasp_settings=OrderedDict(ibrion=1,
                                                 nsw=100,
                                                 isif=7,
                                                 isym=0,
                                                 ediff=1e-8,
                                                 kpts=[10, 10, 10],
                                                 prec='Accurate',
                                                 **settings))


def adsorption_parameters(adsorbate,
                          adsorption_site=None,
                          slabrepeat='(1, 1)',
                          num_slab_atoms=0,
                          settings='beef-vdw'):
    '''
    Generate some default parameters for an adsorption configuration and expected
    relaxation settings
    '''
    if isinstance(settings, str):
        settings = calc_settings(settings)
    adsorbateStructures = {'CO': {'name':'CO', 'atoms': Atoms('CO', positions=[[0., 0., 0.],
                                                                               [0., 0., 1.2]])},
                           'H': {'name': 'H', 'atoms': Atoms('H', positions=[[0., 0., -0.5]])},
                           'O': {'name': 'O', 'atoms': Atoms('O', positions=[[0., 0., 0.]])},
                           'C': {'name': 'C', 'atoms': Atoms('C', positions=[[0., 0., 0.]])},
                           '': {'name': '', 'atoms': Atoms()},
                           'U': {'name': 'U', 'atoms': Atoms('U', positions=[[0., 0., 0.]])},
                           'OH': {'name': 'OH', 'atoms': Atoms('OH', positions=[[0., 0., 0.],
                                                                                [0., 0., 0.96]])},
                           'OOH': {'name': 'OOH', 'atoms': Atoms('OOH', positions=[[0., 0., 0.],
                                                                                   [0., 0., 1.55], 
                                                                                       [0, 0.94, 1.80]])}}

    # This controls how many configurations get submitted if multiple configurations
    # match the criteria
    return OrderedDict(numtosubmit=1,
                       min_xy=4.5,
                       relaxed=True,
                       num_slab_atoms=num_slab_atoms,
                       slabrepeat=slabrepeat,
                       adsorbates=[OrderedDict(name=adsorbate,
                                               atoms=pickle.dumps(adsorbateStructures[adsorbate]['atoms']).encode('hex'),
                                               adsorption_site=adsorption_site)],
                       vasp_settings=OrderedDict(ibrion=2,
                                                 nsw=200,
                                                 isif=0,
                                                 isym=0,
                                                 kpts=[4, 4, 1],
                                                 lreal='Auto',
                                                 ediffg=-0.03,
                                                 **settings))
