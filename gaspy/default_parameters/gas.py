from collections import OrderedDict
from calc_settings import calc_settings


def parameter_gas(gasname, settings='beef-vdw'):
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
