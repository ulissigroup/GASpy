'''
This function is used in various scripts to convert a dictionary of vasp settings into a format
that is acceptable by ase-db.
Input
    vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein may have a
                            different type depending on the VASP setting.
Output
    vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein is either
                            an int, float, boolean, or string.
'''

def vasp_settings_to_str(vasp_settings):
    # Create a local copy to make sure we are not acting on the global copy(?)
    vasp_settings = vasp_settings.copy()

    # For each item in "vasp_settings"...
    for key in vasp_settings:
        # Find anything that's not a string, integer, float, or boolean...
        if not isinstance(vasp_settings[key], (str, int, float, bool)):
            # And turn it into a string
            vasp_settings[key] = str(vasp_settings[key])

    return vasp_settings
