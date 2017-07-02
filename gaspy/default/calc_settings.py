from collections import OrderedDict
from xc_settings import xc_settings

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
