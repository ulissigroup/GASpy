from collections import OrderedDict
import numpy as np
from ..utils.print_dict import print_dict


def running_fireworks(name_dict, launchpad):
    '''
    Return the running, ready, or completed fireworks on the launchpad with a given name
    name_dict   name dictionary to search for
    launchpad   launchpad to use
    '''
    # Make a mongo query
    name = {}
    # Turn a nested dictionary into a series of mongo queries
    for key in name_dict:
        if isinstance(name_dict[key], dict) or isinstance(name_dict[key], OrderedDict):
            for key2 in name_dict[key]:
                name['name.%s.%s'%(key, key2)] = name_dict[key][key2]
        else:
            if key == 'shift':
                # Search for a range of shift parameters up to 4 decimal place
                shift = float(np.round(name_dict[key], 4))
                name['name.%s'%key] = {'$gte':shift-1e-4, '$lte':shift+1e-4}
            else:
                name['name.%s'%key] = name_dict[key]

    # Get all of the fireworks that are completed, running, or ready (i.e., not fizzled
    # or defused.)
    fw_ids = launchpad.get_fw_ids(name)
    fw_list = []
    for fwid in fw_ids:
        fw = launchpad.get_fw_by_id(fwid)
        if fw.state in ['RUNNING', 'COMPLETED', 'READY']:
            fw_list.append(fwid)
    # Return the matching fireworks
    if len(fw_list) == 0:
        print('        No matching FW for:')
        print_dict(name, indent=3)
    return fw_list
