from pprint import pprint
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
# TODO:  Update our version of PyMatGen. This version of the code uses a forked
# PyMatGen with additional capabilities, but the master branch of PyMatGen has since been
# updated. We should start using it.
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from vasp.mongo import MongoDatabase
from defaults import adsorbates_dict


def print_dict(d, indent=0):
    '''
    This function prings a nested dictionary, but in a prettier format. This is strictly for
    debugging purposes.

    Inputs:
        d       The nested dictionary to print
        indent  How many tabs to start the printing at
    '''
    if isinstance(d, dict):
        for key, value in d.iteritems():
            # If the dictionary key is `spec`, then it's going to print out a bunch of
            # messy looking things we don't care about. So skip it.
            if key != 'spec':
                print('\t' * indent + str(key))
                if isinstance(value, dict) or isinstance(value, list):
                    print_dict(value, indent+1)
                else:
                    print('\t' * (indent+1) + str(value))
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, dict) or isinstance(item, list):
                print_dict(item, indent+1)
            else:
                print('\t' * (indent+1) + str(item))
    else:
        pass


def get_aux_db():
    ''' This is the information for the Auxiliary vasp.mongo database '''
    return MongoDatabase(host='mongodb01.nersc.gov',
                         port=27017,
                         user='admin_zu_vaspsurfaces',
                         password='$TPAHPmj',
                         database='vasp_zu_vaspsurfaces',
                         collection='atoms')


def vasp_settings_to_str(vasp_settings):
    '''
    This function is used in various scripts to convert a dictionary of vasp settings
    into a format that is acceptable by ase-db.

    Input:
        vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein
                                may have a different type depending on the VASP setting.
    Output:
        vasp_settings   [dict]  Each key is a VASP setting. Each object contained therein
                                is either an int, float, boolean, or string.
    '''
    vasp_settings = vasp_settings.copy()

    # For each item in "vasp_settings"...
    for key in vasp_settings:
        # Find anything that's not a string, integer, float, or boolean...
        if not isinstance(vasp_settings[key], (str, int, float, bool)):
            # And turn it into a string
            vasp_settings[key] = str(vasp_settings[key])

    return vasp_settings


def ads_dict(adsorbate):
    '''
    This is a helper function to take an adsorbate as a string (e.g. 'CO') and attempt to
    return an atoms object for it, primarily as a way to count the number of constitutent
    atoms in the adsorbate.
    '''
    # Try to create an [atoms class] from the input.
    try:
        atoms = Atoms(adsorbate)
    except ValueError:
        pprint("Not able to create %s with ase.Atoms. Attempting to look in GASpy's dictionary..." \
               % adsorbate)

        # If that doesn't work, then look for the adsorbate in our library of adsorbates
        try:
            atoms = _adsorbates_dict()[adsorbate]
        except KeyError:
            print('%s is not is GASpy library of adsorbates. You need to add it to the adsorbates_dict function in gaspy.defaults' \
                  % adsorbate)

    # Return the atoms
    return atoms


def constrain_slab(atoms, n_ads_atoms=0, z_cutoff=3.):
    '''
    Define a function, "constrain_slab" to impose slab constraints prior to relaxation.
    This function assumes that the indices of the adsorbate atoms come before the slab
    atoms.

    Inputs:
        atoms       ASE-atoms class of the adsorbate + slab system
        n_ads_atoms Number of adsorbate atoms
        z_cutoff    The threshold to see if slab atoms are in the same plane as the
                    highest atom in the slab
    '''
    # Pull out the constraints that may already be on our atoms object.
    constraints = atoms.constraints 

    # Constrain atoms except for the top layer. To do this, we first pull some information out
    # of the atoms object.
    scaled_positions = atoms.get_scaled_positions() #
    z_max = np.max([pos[2] for pos in scaled_positions[n_ads_atoms:]]) # Scaled height of highest slab atom
    z_min = np.min([pos[2] for pos in scaled_positions[n_ads_atoms:]]) # Scaled height of lowest slab atom
    # Add the constraint, which is a binary list (i.e., 1's & 0's) used to identify which atoms
    # to fix or not. The indices of the list correspond to the indices of the atoms in the "atoms".
    if atoms.cell[2, 2] > 0:
        constraints.append(FixAtoms(mask=[pos[2] < z_max-(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    else:
        constraints.append(FixAtoms(mask=[pos[2] > z_min+(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))

    # Enact the constraints on the local atoms instance
    atoms.set_constraint(constraints)
    return atoms


def fingerprint_atoms(atoms, siteind):
    '''
    This function is used to fingerprint an atoms object, where the "fingerprint" is a dictionary
    of properties that we believe may be adsorption motifs.

    Inputs:
        atoms       atoms object to fingerprint
        siteind     the position of the binding atom in the adsorbate (assumed to be the first atom
                    of the adsorbate)
    '''
    # Delete the adsorbate except for the binding atom, then turn it into a uranium atom so
    # we can keep track of it in the coordination calculation
    atoms = atoms[0:siteind+1]
    atoms[-1].symbol = 'U'

    # Turn the atoms into a pymatgen structure file
    struct = AseAtomsAdaptor.get_structure(atoms)
    # PyMatGen [vcf class] of our system
    vcf = VoronoiCoordFinder(struct)
    # [list] of PyMatGen [periodic site class]es for each of the atoms that are
    # coordinated with the adsorbate
    coordinated_atoms = vcf.get_coordinated_sites(siteind, 0.8)
    # The elemental symbols for all of the coordinated atoms in a [list] of [unicode] objects
    coordinated_symbols = map(lambda x: x.species_string, coordinated_atoms)
    # Take out atoms that we assume are not part of the slab
    coordinated_symbols = [a for a in coordinated_symbols if a not in ['U']]
    # Turn the [list] of [unicode] values into a single [unicode]
    coordination = '-'.join(sorted(coordinated_symbols))

    # Make a [list] of human-readable coordination sites [unicode] for all of the slab atoms
    # that are coordinated to the adsorbate, "neighborcoord"
    neighborcoord = []
    for i in coordinated_atoms:
        # [int] that yields the slab+ads system's atomic index for the 1st-tier-coordinated atom
        neighborind = [site[0] for site in enumerate(struct) if i.distance(site[1]) < 0.1][0]
        # [list] of PyMatGen [periodic site class]es for each of the atoms that are coordinated
        # with the adsorbate
        coord = vcf.get_coordinated_sites(neighborind, 0.2)
        # The elemental symbols for all of the 2nd-tier-coordinated atoms in a [list] of
        # [unicode] objects
        coord_symbols = map(lambda x: x.species_string, coord)
        # Take out atoms that we assume are not part of the slab
        coord_symbols = [a for a in coord_symbols if a not in ['U']]
        # Sort the list of 2nd-tier-coordinated atoms to induce consistency
        coord_symbols.sort()
        # Turn the [list] of [unicode] values into a single [unicode]
        neighborcoord.append(i.species_string+':'+'-'.join(coord_symbols))

    # [list] of PyMatGen [periodic site class]es for each of the atoms that are
    # coordinated with the adsorbate
    coordinated_atoms_nextnearest = vcf.get_coordinated_sites(siteind, 0.2)
    # The elemental symbols for all of the coordinated atoms in a [list] of [unicode] objects
    coordinated_symbols_nextnearest = map(lambda x: x.species_string,
                                          coordinated_atoms_nextnearest)
    # Take out atoms that we assume are not part of the slab
    coordinated_symbols_nextnearest = [a for a in coordinated_symbols_nextnearest
                                       if a not in ['U']]
    # Turn the [list] of [unicode] values into a single [unicode]
    coordination_nextnearest = '-'.join(sorted(coordinated_symbols_nextnearest))

    # Return a dictionary with each of the fingerprints. Any key/value pair can be added here
    # and will propagate up the chain
    return {'coordination':coordination,
            'neighborcoord':neighborcoord,
            'natoms':len(atoms),
            'nextnearestcoordination':coordination_nextnearest}


def _label_structure_with_surface(slabAtoms, bulkAtoms):
    '''
    This script/function calculates possible adsorption sites of a slab of atoms. It is
    used primarily as a helper function for the `find_adsorption_sites` function, thus
    the leading `_`

    Inputs:
        slabAtoms   [atoms class]   The slab where you are trying to find adsorption sites
        bulkAtoms   [atoms class]   The original bulk crystal structure of the slab
    Outputs:
        slab_struct
    '''
    # Convert the slab and bulk from [atoms class]es to a
    # [structure class]es (i.e., from ASE format to PyMatGen format)
    slab_struct = AseAtomsAdaptor.get_structure(slabAtoms)
    bulk_struct = AseAtomsAdaptor.get_structure(bulkAtoms)

    # Create [VCF class]es for the slab and bulk, which are PyMatGen class that may
    # be used to find adsorption sites
    vcf_surface = VoronoiCoordFinder(slab_struct)
    vcf_bulk = VoronoiCoordFinder(bulk_struct)

    # Get the chemical formula
    formula = np.unique(slabAtoms.get_chemical_symbols())
    # Initialize the keys for "cn_el" [dict], which will hold the [list of int] coordination
    # numbers of each of atom in the bulk, with the key being the element. Coordination
    # numbers of atoms that share an element with another atom in the bulk are addended to
    # each element's list.
    cn_el = {}
    for element in formula:
        cn_el[element] = []
    # For each atom in the bulk, calculate the coordination number [int] and store it in "cn_el"
    # [dict where key=element [str]]
    for i, atom in enumerate(bulkAtoms):
        # Fetch the atomic symbol of the element, "el" [str]
        el = str(bulk_struct[i].specie)
        # Use PyMatGen to identify the "coordinated_neighbors" [list of classes]
        # Note that we use a tolerance of 0.1 to be consistent with PyMatGen
        coordinated_neighbors = vcf_bulk.get_coordinated_sites(i, tol=0.1)
        # Calculate the number of coordinated neighbors [int]
        num_neighbors = len(coordinated_neighbors)
        # Store this number in "cn_el" [dict]. Note that cn_el[el] will return a list of
        # coordination numbers for each atom whose element matches the "el" key.
        cn_el[el].append(num_neighbors)
    # Calculate "mean_cn_el" [dict], which will hold the mean coordination number [float] of
    # each element in the bulk
    mean_cn_el = {}
    for element in formula:
        #mean_cn_el[element] = float(sum(cn_el[element]))/len(cn_el[element])
        mean_cn_el[element] = sum(cn_el[element])/len(cn_el[element])

    # Calculate "average_z" [float], the mean z-level of all the atoms in the slab
    average_z = np.average(slab_struct.cart_coords[:, -1])

    # Initialize a couple of [list] objects that we will pass to PyMatGen later
    cn_surf = []
    plate_surf = []
    # For each atom in the slab, we calculate the coordination number and then determine whether
    # or not the atom is on the surface.
    for i, atom in enumerate(slab_struct):
        # "cn_surf" [list of floats] holds the coordination numbers of the atoms in the slab.
        # Note that we use a tolerance of 0.2 instead of 0.1. This may improve the scripts
        # ability to identify adsorption sites.
        cn_surf.append(len(vcf_surface.get_coordinated_sites(i, tol=0.2)))
        # Given this atom's element, we fetch the mean coordination number of the same element,
        # but in the bulk structure instead of the slab structure. "cn_Bulk" is a [float].
        cn_Bulk = mean_cn_el[str(slab_struct[i].specie)]
        # If the coordination number of the atom changes between the slab and bulk structures
        # AND if the atom is above the centerline of the slab...
        if cn_surf[-1] != cn_Bulk and atom.coords[-1] > average_z:
            # then the atom is labeled as a "surface" atom...
            plate_surf.append('surface')
        else:
            # else it is a subsurface atom. Note that "plate_surf" is a [list of str].
            plate_surf.append('subsurface')

    # We add "new_site_properties" to "slab_struct" [PyMatGen structure class]
    new_site_properties = {'surface_properties':plate_surf, 'coord':cn_surf}
    slab_struct = slab_struct.copy(site_properties=new_site_properties)

    return slab_struct


def find_adsorption_sites(slabAtoms, bulkAtoms):
    '''
    This script/function calculates possible adsorption sites of a slab of atoms
    
    Inputs:
        slabAtoms   [atoms class]   The slab where you are trying to find adsorption sites
        bulkAtoms   [atoms class]   The original bulk crystal structure of the slab
    Outputs:
        sites       [list]  A list of [array]s, which contain the x-y-z coordinates
                            [floats] of the adsorptions sites.
    '''
    slab_struct = _label_structure_with_surface(slabAtoms, bulkAtoms)
    # Finally, we call "AdsorbateSiteFinder", which is a function in a branch of PyMatGen,
    # to create "asf" [class]
    asf = AdsorbateSiteFinder(slab_struct)
    # Then we use "asf" [class] to calculate "sites" [list of arrays of floats], which holds
    # the cartesion coordinates for each of the adsorption sites.
    sites = asf.find_adsorption_sites()

    return sites


# TODO:  This is not currently used. If it stays that way long enough, we should get rid of it.
def calculate_top(atoms, num_adsorbate_atoms=0):
    ''' We use this function to determine which side is the "top" side '''
    if num_adsorbate_atoms > 0:
        atoms = atoms[0:-num_adsorbate_atoms]
    zpos = atoms.positions[:, 2]
    return np.sum((zpos-zpos.mean())*atoms.get_masses()) > 0