"""
This script/function calculates possible adsorption sites of a slab of atoms
Inputs:
  slabAtoms   [atoms class]   The slab where you are trying to find adsorption sites
  bulkAtoms   [atoms class]   The original bulk crystal structure of the slab
Outputs:
  sites       [list]          A list of [array]s, which contain the x-y-z coordinates [floats] of
                              the adsorptions sites.
"""

# Import the necessary modules. Note: This script uses the "modifyAdsorption" branch of PyMatGen.
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

# This function must be supplied with both "slabAtoms" and "bulkAtoms" (see above)
def find_adsorption_sites(slabAtoms, bulkAtoms):
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
    # numbers of each of atom in the bulk, with the key being the element. Coordination numbers of
    # atoms that share an element with another atom in the bulk are addended to each element's list.
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
    # Calculate "mean_cn_el" [dict], which will hold the mean coordination number [float] of each
    # element in the bulk
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
        # Note that we use a tolerance of 0.2 instead of 0.1. This may improve the scripts ability
        # to identify adsorption sites.
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

    # Finally, we call "AdsorbateSiteFinder", which is a function in a branch of PyMatGen,
    # to create "asf" [class]
    asf = AdsorbateSiteFinder(slab_struct)
    # Then we use "asf" [class] to calculate "sites" [list of arrays of floats], which holds
    # the cartesion coordinates for each of the adsorption sites.
    sites = asf.find_adsorption_sites()

    return sites
