from collections import OrderedDict
import cPickle as pickle
from ase import Atoms
from calc_settings import calc_settings

def adsorption(adsorbate,
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
    adsorbateStructures = {'CO':{'atoms':Atoms('CO', positions=[[0.,0.,0.],[0.,0.,1.2]]),  'name':'CO'},
                           'H':{'atoms':Atoms('H',   positions=[[0.,0.,-0.5]]),            'name':'H'},
                           'O':{'atoms':Atoms('O',   positions=[[0.,0.,0.]]),              'name':'O'},
                           'C':{'atoms':Atoms('C',   positions=[[0.,0.,0.]]),              'name':'C'},
                           '':{'atoms':Atoms(),                                            'name':''},
                           'U':{'atoms':Atoms('U',   positions=[[0.,0.,0.]]),              'name':'U'},
                           'OH':{'atoms':Atoms('OH', positions=[[0.,0.,0.],[0.,0.,0.96]]), 'name':'OH'}}

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
