''' Various functions that may be used across GASpy and its submodules '''

import pdb
import warnings
from pprint import pprint
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from vasp.mongo import MongoDatabase
from . import defaults


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


def get_adsorption_db():
    ''' This is the information for the Adsorption Energy vasp.mongo database '''
    return MongoDatabase(host='mongodb01.nersc.gov',
                         port=27017,
                         user='admin_zu_vaspsurfaces',
                         password='$TPAHPmj',
                         database='vasp_zu_vaspsurfaces',
                         collection='adsorption')


def get_catalog_db():
    ''' This is the information for the Adsorption Site Catalog vasp.mongo database '''
    return MongoDatabase(host='mongodb01.nersc.gov',
                         port=27017,
                         user='admin_zu_vaspsurfaces',
                         password='$TPAHPmj',
                         database='vasp_zu_vaspsurfaces',
                         collection='catalog')


def get_docs(client=get_adsorption_db(), collection_name='adsorption', fingerprints=None,
             adsorbates=None, calc_settings=None, vasp_settings=None,
             energy_min=None, energy_max=None, f_max=None,
             ads_move_max=None, bare_slab_move_max=None, slab_move_max=None):
    # pylint: disable=too-many-branches, too-many-arguments
    '''
    This function uses a mongo aggregator to find unique mongo docs and then returns them
    in two different forms:  a "raw" form (list of dicts) and a "parsed" form (dict of lists).
    Note that since we use a mongo aggregator, this function will return only unique mongo
    docs (as per the fingerprints supplied by the user); do not expect a mongo doc per
    matching database entry.

    Inputs:
        client              Mongo client object
        collection_name     The collection name within the client that you want to look at
        fingerprints        A dictionary of fingerprints and their locations in our
                            mongo documents. For example:
                                fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                                                'coordination': '$processed_data.fp_init.coordination'}
                            If `None`, then pull the default set of fingerprints.
        adsorbates          A list of adsorbates that you want to find matches for
        calc_settings       An optional argument that will only pull out data with these
                            calc settings (e.g., 'beef-vdw' or 'rpbe').
        vasp_settings       An optional argument that will only pull out data with these
                            vasp settings. Any assignments to `gga` here will overwrite
                            `calc_settings`.
        energy_min          The minimum adsorption energy to pull from the Local DB (eV)
        energy_max          The maximum adsorption energy to pull from the Local DB (eV)
        ads_move_max        The maximum distance that an adsorbate atom may move (angstrom)
        bare_slab_move_max  The maxmimum distance that a slab atom may move when it is relaxed
                            without an adsorbate (angstrom)
        slab_move_max       The maximum distance that a slab atom may move (angstrom)
        f_max               The upper limit on the maximum force on an atom in the system
    Output:
        docs    Mongo docs; a list of dictionaries for each database entry
        p_docs  "parsed docs"; a dictionary whose keys are the keys of `fingerprints` and
                whose values are lists of the results returned by each query within
                `fingerprints`.
    '''
    if not fingerprints:
        fingerprints = defaults.fingerprints()

    # Initialize
    p_docs = dict.fromkeys(fingerprints)
    # Put the "fingerprinting" into a `group` dictionary, which we will
    # use to pull out data from the mongo database. Also, initialize
    # a `match` dictionary, which we will use to filter results.
    group = {'$group': {'_id': fingerprints}}
    match = {'$match': {}}

    # Create `match` filters to search by. Use if/then statements to create the filters
    # only if the user specifies them.
    if not calc_settings:
        pass
    elif calc_settings == 'rpbe':
        match['$match']['processed_data.vasp_settings.gga'] = 'RP'
    elif calc_settings == 'beef-vdw':
        match['$match']['processed_data.vasp_settings.gga'] = 'BF'
    else:
        raise Exception('Unknown calc_settings')
    if vasp_settings:
        for key, value in vasp_settings.iteritems():
            match['$match']['processed_data.vasp_settings.%s' % key] = value
        # Alert the user that they tried to specify the gga twice.
        if ('gga' in vasp_settings and calc_settings):
            warnings.warn('User specified both calc_settings and vasp_settings.gga. GASpy will default to the given vasp_settings.gga', SyntaxWarning)
    if adsorbates:
        match['$match']['processed_data.calculation_info.adsorbate_names'] = adsorbates
    # Multi-conditional for the energy for the different ways a user can define
    # energy constraints
    if (energy_max and energy_min):
        match['$match']['results.energy'] = {'$gt': energy_min, '$lt': energy_max}
    elif (energy_max and not energy_min):
        match['$match']['results.energy'] = {'$lt': energy_max}
    elif (not energy_max and energy_min):
        match['$match']['results.energy'] = {'$gt': energy_min}
    # We do a doubly-nested element match because `results.forces` is a doubly-nested
    # list of forces (1st layer is atoms, 2nd layer is cartesian directions, final
    # layer is the forces on that atom in that direction).
    if f_max:
        match['$match']['results.forces'] = {'$not': {'$elemMatch': {'$elemMatch': {'$gt': f_max}}}}
    if ads_move_max:
        match['$match']['processed_data.movement_data.max_adsorbate_movement'] = \
                {'$lt': ads_move_max}
    if bare_slab_move_max:
        match['$match']['processed_data.movement_data.max_bare_slab_movement'] = \
                {'$lt': bare_slab_move_max}
    if slab_move_max:
        match['$match']['processed_data.movement_data.max_surface_movement'] = \
                {'$lt': slab_move_max}

    # Compile the pipeline; add matches only if any matches are specified
    if match['$match']:
        pipeline = [match, group]
    else:
        pipeline = [group]

    # Get the particular collection from the mongo client's database
    collection = getattr(client.db, collection_name)
    # Create the cursor. We set allowDiskUse=True to allow mongo to write to
    # temporary files, which it needs to do for large databases. We also
    # set useCursor=True so that `aggregate` returns a cursor object
    # (otherwise we run into memory issues).
    cursor = collection.aggregate(pipeline, allowDiskUse=True, useCursor=True)
    # Use the cursor to pull all of the information we want out of the database, and
    # then parse it.
    docs = [doc['_id'] for doc in cursor]
    for key in p_docs.keys():
        p_docs[key] = [doc[key] if key in doc else None for doc in docs]
    return docs, p_docs


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
            atoms = defaults.adsorbates_dict()[adsorbate]
        except KeyError:
            print('%s is not is GASpy library of adsorbates. You need to add it to the adsorbates_dict function in gaspy.defaults' \
                  % adsorbate)

    # Return the atoms
    return atoms


def remove_adsorbate(adslab):
    '''
    This sub-function removes adsorbates from an adslab.

    Input:
        adslab  The ase.Atoms object of the adslab. The adsorbate atom(s) must be tagged
                with non-zero integers, while the slab atoms must be tagged with zeroes.
    Outputs:
        slab                The ase.Atoms object of the bare slab. It will not have any
                            constraints of the input `atoms`.
        binding_positions   A dictionary whose keys are the tags of the adsorbates and whose
                            values are the cartesian coordinates of the binding site.
    '''
    # We need to make a local copy of the atoms so that when we start manipulating it here,
    # the changes do not propagate.
    slab = adslab.copy()
    # ASE doesn't like deleting atoms with constraints on them, so we get rid of them.
    slab.set_constraint()
    # Pull out the tags so we know what to delete
    tags = slab.get_tags()
    # Initialize `binding_positions`. We delete the "zero" tag because that's for slabs.
    binding_positions = dict.fromkeys(tags)
    del binding_positions[0]
    # `atoms_range` is a list of atom indices for `slab`. We initialize it because
    # we need to reverse it before iterating through (to maintain proper indexing).
    atoms_range = range(0, len(slab))
    atoms_range.reverse()

    # Delete the adsorbates while simultaneously storing the position of the binding atom.
    for i in atoms_range:
        if tags[i]:
            # Note that we are continuously overwrite the binding positions. The last
            # entry/write # corresponds to the position of the "first" atom for each adsorbate,
            # thus the requirement described in this function's docstring.
            binding_positions[tags[i]] = slab.get_positions()[i]
            del slab[i]

    return slab, binding_positions


def constrain_slab(atoms, z_cutoff=3.):
    '''
    Define a function, "constrain_slab" to impose slab constraints prior to relaxation.

    Inputs:
        atoms       ASE-atoms class of the adsorbate + slab system. The tags of these
                    atoms must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
        z_cutoff    The threshold to see if slab atoms are in the same plane as the
                    highest atom in the slab
    '''
    # Remove the adsorbate so that we can find the number of slab atoms
    slab, binding_positions = remove_adsorbate(atoms)
    n_slab_atoms = len(slab)

    # Find the scaled height of the highest and lowest slab atoms.
    scaled_positions = atoms.get_scaled_positions()
    # In an old version of GASpy, we added the adsorbate to the slab instead of adding
    # the slab to the adsorbate. These conditionals determine if the adslab is old or
    # not and processes it accordingly.
    if atoms[0].tag:
        z_max = np.max([pos[2] for pos in scaled_positions[-n_slab_atoms:]])
        z_min = np.min([pos[2] for pos in scaled_positions[-n_slab_atoms:]])
    else:
        z_max = np.max([pos[2] for pos in scaled_positions[:n_slab_atoms]])
        z_min = np.min([pos[2] for pos in scaled_positions[:n_slab_atoms]])

    # Use the scaled heights to fix any atoms below the surfaces of the slab
    constraints = atoms.constraints
    if atoms.cell[2, 2] > 0:
        constraints.append(FixAtoms(mask=[pos[2] < z_max-(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    else:
        constraints.append(FixAtoms(mask=[pos[2] > z_min+(z_cutoff/np.linalg.norm(atoms.cell[2]))
                                          for pos in scaled_positions]))
    atoms.set_constraint(constraints)

    return atoms


def fingerprint_atoms(atoms):
    '''
    This function is used to fingerprint an adslabs atoms object, where the "fingerprint" is a
    dictionary of properties that we believe may be adsorption motifs.

    Inputs:
        atoms   Atoms object to fingerprint. The slab atoms must be tagged with 0 and
                adsorbate atoms must be tagged with non-zero integers. This function also
                assumes that the first atom in each adsorbate is the binding atom (e.g.,
                of all atoms with tag==1, the first atom is the binding; the same goes for
                tag==2 and tag==3 etc.).
    '''
    # Remove the adsorbate(s) while finding the binding position(s)
    atoms, binding_positions = remove_adsorbate(atoms)
    # Add Uranium atoms at each of the binding sites so that we can use them for fingerprinting.
    for tag in binding_positions:
        atoms += Atoms('U', positions=[binding_positions[tag]])

    # Turn the atoms into a pymatgen structure object so that we can use the VCF to find
    # the coordinated sites.
    struct = AseAtomsAdaptor.get_structure(atoms)
    vcf = VoronoiCoordFinder(struct, allow_pathological=True)
    coordinated_atoms = vcf.get_coordinated_sites(len(atoms)-1, 0.8)
    # Create a list of symbols of the coordinations, remove uranium from the list, and
    # then turn the list into a single, human-readable string.
    coordinated_symbols = map(lambda x: x.species_string, coordinated_atoms)
    coordinated_symbols = [a for a in coordinated_symbols if a not in ['U']]
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
    coordinated_atoms_nextnearest = vcf.get_coordinated_sites(0, 0.2)
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


def _label_structure_with_surface(slabAtoms, bulkAtoms, height_threshold=3.):
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
    vcf_surface = VoronoiCoordFinder(slab_struct, allow_pathological=True)
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
        coordinated_neighbors = vcf_bulk.get_coordinated_sites(i, tol=0.4)
        # Calculate the number of coordinated neighbors [int]
        num_neighbors = len(coordinated_neighbors)
        # Store this number in "cn_el" [dict]. Note that cn_el[el] will return a list of
        # coordination numbers for each atom whose element matches the "el" key.
        cn_el[el].append(num_neighbors)
    # Calculate "mean_cn_el" [dict], which will hold the mean coordination number [float] of
    # each element in the bulk
    mean_cn_el = {}
    min_cn_el = {}
    for element in formula:
        #mean_cn_el[element] = float(sum(cn_el[element]))/len(cn_el[element])
        mean_cn_el[element] = sum(cn_el[element])/len(cn_el[element])
        min_cn_el[element] = np.min(cn_el[element])

    # Calculate "average_z" [float], the mean z-level of all the atoms in the slab
    average_z = np.average(slab_struct.cart_coords[:, -1])
    max_z = np.max(slab_struct.cart_coords[:, -1])

    # Initialize a couple of [list] objects that we will pass to PyMatGen later
    cn_surf = []
    plate_surf = []
    # For each atom in the slab, we calculate the coordination number and then determine whether
    # or not the atom is on the surface.

    for i, atom in enumerate(slab_struct):
        # "cn_surf" [list of floats] holds the coordination numbers of the atoms in the slab.
        # Note that we use a tolerance of 0.2 instead of 0.1. This may improve the scripts
        # ability to identify adsorption sites.
        cn_surf.append(len(vcf_surface.get_coordinated_sites(i, tol=0.4)))
        # Given this atom's element, we fetch the mean coordination number of the same element,
        # but in the bulk structure instead of the slab structure. "cn_Bulk" is a [float].
        element = str(slab_struct[i].specie)
        cn_Bulk = mean_cn_el[element]
        # If the coordination number of the atom changes between the slab and bulk structures
        # AND if the atom is above the centerline of the slab...
        if (cn_surf[-1]<min_cn_el[element] and atom.coords[-1] > average_z and atom.coords[-1]>max_z-height_threshold) or atom.coords[-1]>max_z-1.:
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
    sitedict = asf.find_adsorption_sites(z_oriented=True, put_inside=True)
    # Some versions of PyMatGen provide an `all` key. If it's there, then just pull out
    # all the sites that way. If not, then pull out all the sites manually.
    try:
        sites = sitedict['all']
    except KeyError:
        sites = []
        for key, value in sitedict.iteritems():
            sites += value
    return sites


def find_max_movement(atoms_initial, atoms_final):
    '''
    Given ase.Atoms objects, find the furthest distance that any single atom in a set of
    atoms traveled (in Angstroms)

    Inputs:
        initial_atoms   ase.Atoms objects in their initial state
        final_atoms     ase.Atoms objects in their final state
    '''
    # Calculate the distances for each atom
    distances = atoms_final.positions - atoms_initial.positions

    # Reduce the distances in case atoms wrapped around (the minimum image convention)
    dist, Dlen = find_mic(distances, atoms_final.cell, atoms_final.pbc)

    return np.max(Dlen)
