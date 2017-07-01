from collections import OrderedDict
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
