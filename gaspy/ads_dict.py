from pprint import pprint
from ase import Atoms

def ads_dict(adsorbate):
    '''
    This is a helper function to take an adsorbate as a string (e.g. 'CO') and attempt to
    return an atoms object for it, primarily as a way to count the number of constitutent
    atoms in the adsorbate. It also contains a skeleton for the user to manually add atoms
    objects to "atom_dict".
    '''
    # First, add the manually-added adsorbates to the atom_dict lookup variable. Note that
    # 'H' is just an example. It won't actually be used here.
    atom_dict = {'H': Atoms('H')}

    # Try to create an [atoms class] from the input.
    try:
        atoms = Atoms(adsorbate)
    except ValueError:
        pprint("Not able to create %s with ase.Atoms. Attempting to look in GASpy's dictionary..."\
              % adsorbate)

        # If that doesn't work, then look for the adsorbate in the "atom_dict" object
        try:
            atoms = atom_dict[adsorbate]
        except KeyError:
            print('%s is not is GASpy dictionary. You need to construct it manually and add it to \
                  the ads_dict function in gaspy_toolbox.py' % adsorbate)

    # Return the atoms
    return atoms
