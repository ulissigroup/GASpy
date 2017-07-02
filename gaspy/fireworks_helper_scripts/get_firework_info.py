import ase.io
from vasp import Vasp
from ..utils.vasp_settings_to_str import vasp_settings_to_str
from atoms_hex_to_file import atoms_hex_to_file


def get_firework_info(fw):
    '''
    Given a Fireworks ID, this function will return the "atoms" [class] and
    "vasp_settings" [str] used to perform the relaxation
    '''
    atomshex = fw.launches[-1].action.stored_data['opt_results'][1]
    atoms_hex_to_file('atom_temp.traj', atomshex)
    atoms = ase.io.read('atom_temp.traj')
    starting_atoms = ase.io.read('atom_temp.traj', index=0)
    vasp_settings = fw.name['vasp_settings']
    # Guess the pseudotential version if it's not present
    if 'pp_version' not in vasp_settings:
        if 'arjuna' in fw.launches[-1].fworker.name:
            vasp_settings['pp_version'] = '5.4'
        else:
            vasp_settings['pp_version'] = '5.3.5'
        vasp_settings['pp_guessed'] = True
    if 'gga' not in vasp_settings:
        settings = Vasp.xc_defaults[vasp_settings['xc']]
        for key in settings:
            vasp_settings[key] = settings[key]
    vasp_settings = vasp_settings_to_str(vasp_settings)
    return atoms, starting_atoms, atomshex, vasp_settings
