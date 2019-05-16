'''
This submodule contains various functions that operate on both `ase.Atoms`
objects and `pymatgen.Structure` objectsto do various things.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import math
import numpy as np
from scipy.spatial.qhull import QhullError
from ase import Atoms
from ase.build import rotate
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import VoronoiNN
from .utils import unfreeze_dict
from .defaults import slab_settings


def make_slabs_from_bulk_atoms(atoms, miller_indices,
                               slab_generator_settings, get_slab_settings):
    '''
    Use pymatgen to enumerate the slabs from a bulk.

    Args:
        atoms                   The `ase.Atoms` object of the bulk that you
                                want to make slabs out of
        miller_indices          A 3-tuple of integers containing the three
                                Miller indices of the slab[s] you want to
                                make.
        slab_generator_settings A dictionary containing the settings to be
                                passed to pymatgen's `SpaceGroupAnalyzer`
                                class.
        get_slab_settings       A dictionary containing the settings to be
                                ppassed to the `get_slab` method of
                                pymatgen's `SpaceGroupAnalyzer` class.
    Returns:
        slabs   A list of the slabs in the form of pymatgen.Structure
                objects. Note that there may be multiple slabs because
                of different shifts/terminations.
    '''
    # Get rid of the `miller_index` argument, which is superceded by the
    # `miller_indices` argument.
    try:
        slab_generator_settings = unfreeze_dict(slab_generator_settings)
        slab_generator_settings.pop('miller_index')
        warnings.warn('You passed a `miller_index` object into the '
                      '`slab_generator_settings` argument for the '
                      '`make_slabs_from_bulk_atoms` function. By design, '
                      'this function will instead use the explicit '
                      'argument, `miller_indices`.', SyntaxWarning)
    except KeyError:
        pass

    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    struct_stdrd = sga.get_conventional_standard_structure()
    slab_gen = SlabGenerator(initial_structure=struct_stdrd,
                             miller_index=miller_indices,
                             **slab_generator_settings)
    slabs = slab_gen.get_slabs(**get_slab_settings)
    return slabs


def orient_atoms_upwards(atoms):
    '''
    Orient an `ase.Atoms` object upwards so that the normal direction of the
    surface points in the upwards z direction.

    Arg:
        atoms   An `ase.Atoms` object
    Returns:
        atoms   The same `ase.Atoms` object that was input as an argument,
                except the z-direction should be pointing upwards.
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    rotate(atoms,
           atoms.cell[2], (0, 0, 1),    # Point the z-direction upwards
           atoms.cell[0], (1, 0, 0),    # Point the x-direction forwards
           rotate_cell=True)
    return atoms


def constrain_slab(atoms, z_cutoff=3.):
    '''
    This function fixes sub-surface atoms of a slab. Also works on systems that
    have slabs + adsorbate(s), as long as the slab atoms are tagged with `0`
    and the adsorbate atoms are tagged with positive integers.

    Inputs:
        atoms       ASE-atoms class of the slab system. The tags of these atoms
                    must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
        z_cutoff    The threshold to see if slab atoms are in the same plane as
                    the highest atom in the slab
    Returns:
        atoms   A deep copy of the `atoms` argument, but where the appropriate
                atoms are constrained
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = []

    # If the slab is pointing upwards, then fix atoms that are below the
    # threshold
    if atoms.cell[2, 2] > 0:
        max_height = max(atom.position[2] for atom in atoms if atom.tag == 0)
        threshold = max_height - z_cutoff
        for atom in atoms:
            if atom.tag == 0 and atom.position[2] < threshold:
                mask.append(True)
            else:
                mask.append(False)

    # If the slab is pointing downwards, then fix atoms that are above the
    # threshold
    elif atoms.cell[2, 2] < 0:
        min_height = min(atom.position[2] for atom in atoms if atom.tag == 0)
        threshold = min_height + z_cutoff
        for atom in atoms:
            if atom.tag == 0 and atom.position[2] > threshold:
                mask.append(True)
            else:
                mask.append(False)

    else:
        raise RuntimeError('Tried to constrain a slab that points in neither '
                           'the positive nor negative z directions, so we do '
                           'not know which side to fix')

    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms


def is_structure_invertible(structure):
    '''
    This function figures out whether or not an `pymatgen.Structure` object is
    invertible in the z-direction (i.e., is it symmetric about the x-y axis?).

    Arg:
        structure   A `pymatgen.Structure` object.
    Returns
        A boolean indicating whether or not your `ase.Atoms` object is
        invertible.
    '''
    # If any of the operations involve a transformation in the z-direction,
    # then the structure is invertible.
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    for operation in sga.get_symmetry_operations():
        xform_matrix = operation.affine_matrix
        z_xform = xform_matrix[2, 2]
        if z_xform == -1:
            return True

    return False


def flip_atoms(atoms):
    '''
    Flips an atoms object upside down. Normally used to flip slabs.

    Arg:
        atoms   `ase.Atoms` object
    Returns:
        atoms   The same `ase.Atoms` object that was fed as an argument,
                but flipped upside down.
    '''
    atoms = atoms.copy()

    # This is black magic wizardry to me. Good look figuring it out.
    atoms.wrap()
    atoms.rotate(180, 'x', rotate_cell=True, center='COM')
    if atoms.cell[2][2] < 0.:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.wrap()

    return atoms


def tile_atoms(atoms, min_x, min_y):
    '''
    This function will repeat an atoms structure in the x and y direction until
    the x and y dimensions are at least as wide as the given parameters.

    Args:
        atoms   `ase.Atoms` object of the structure that you want to tile
        min_x   The minimum width you want in the x-direction (Angstroms)
        min_y   The minimum width you want in the y-direction (Angstroms)
    Returns:
        atoms_tiled     An `ase.Atoms` object that's just a tiled version of
                        the `atoms` argument.
        (nx, ny)        A 2-tuple containing integers for the number of times
                        the original atoms object was repeated in the x
                        direction and y direction, respectively.
    '''
    x_length = np.linalg.norm(atoms.cell[0])
    y_length = np.linalg.norm(atoms.cell[1])
    nx = int(math.ceil(min_x/x_length))
    ny = int(math.ceil(min_y/y_length))
    n_xyz = (nx, ny, 1)
    atoms_tiled = atoms.repeat(n_xyz)
    return atoms_tiled, (nx, ny)


def find_adsorption_sites(atoms):
    '''
    A wrapper for pymatgen to get all of the adsorption sites of a slab.

    Arg:
        atoms   The slab where you are trying to find adsorption sites in
                `ase.Atoms` format
    Output:
        sites   A `numpy.ndarray` object that contains the x-y-z coordinates of
                the adsorptions sites
    '''
    struct = AseAtomsAdaptor.get_structure(atoms)
    sites_dict = AdsorbateSiteFinder(struct).find_adsorption_sites(put_inside=True)
    sites = sites_dict['all']
    return sites


def add_adsorbate_onto_slab(adsorbate, slab, site):
    '''
    There are a lot of small details that need to be considered when adding an
    adsorbate onto a slab. This function will take care of those details for
    you.

    Args:
        adsorbate   An `ase.Atoms` object of the adsorbate
        slab        An `ase.Atoms` object of the slab
        site        A 3-long sequence containing floats that indicate the
                    cartesian coordinates of the site you want to add the
                    adsorbate onto.
    Returns:
        adslab  An `ase.Atoms` object containing the slab and adsorbate.
                The sub-surface slab atoms will be fixed, and all adsorbate
                constraints should be preserved. Slab atoms will be tagged
                with a `0` and adsorbate atoms will be tagged with a `1`.
    '''
    adsorbate = adsorbate.copy()    # To make sure we don't mess with the original
    adsorbate.translate(site)

    adslab = adsorbate + slab
    adslab.cell = slab.cell
    adslab.pbc = [True, True, True]

    # We set the tags of slab atoms to 0, and set the tags of the adsorbate to 1.
    # In future version of GASpy, we intend to set the tags of co-adsorbates
    # to 2, 3, 4... etc (per co-adsorbate)
    tags = [1]*len(adsorbate)
    tags.extend([0]*len(slab))
    adslab.set_tags(tags)

    # Fix the sub-surface atoms
    adslab_constrained = constrain_slab(adslab)

    return adslab_constrained


def fingerprint_adslab(atoms):
    '''
    This function will fingerprint a slab+adsorbate atoms object for you.
    Currently, it only works with one adsorbate.

    Arg:
        atoms   `ase.Atoms` object to fingerprint. The slab atoms must be
                tagged with 0 and adsorbate atoms must be tagged with
                non-zero integers.  This function also assumes that the
                first atom in each adsorbate is the binding atom (e.g.,
                of all atoms with tag==1, the first atom is the binding;
                the same goes for tag==2 and tag==3 etc.).
    Returns:
        fingerprint A dictionary whose keys are:
                        coordination            A string indicating the
                                                first shell of
                                                coordinated atoms
                        neighborcoord           A list of strings
                                                indicating the coordination
                                                of each of the atoms in
                                                the first shell of
                                                coordinated atoms
                        nextnearestcoordination A string identifying the
                                                coordination of the
                                                adsorbate when using a
                                                loose tolerance for
                                                identifying "neighbors"
    '''
    # Replace the adsorbate[s] with a single Uranium atom at the first binding
    # site. We need the Uranium there so that pymatgen can find its
    # coordination.
    atoms, binding_positions = remove_adsorbate(atoms)
    atoms += Atoms('U', positions=[binding_positions[1]])
    uranium_index = atoms.get_chemical_symbols().index('U')
    struct = AseAtomsAdaptor.get_structure(atoms)
    try:
        # We have a standard and a loose Voronoi neighbor finder for various
        # purposes
        vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)
        vnn_loose = VoronoiNN(allow_pathological=True, tol=0.2, cutoff=10)

        # Find the coordination
        nn_info = vnn.get_nn_info(struct, n=uranium_index)
        coordination = __get_coordination_string(nn_info)

        # Find the neighborcoord
        neighborcoord = []
        for neighbor_info in nn_info:
            # Get the coordination of this neighbor atom, e.g., 'Cu-Cu'
            neighbor_index = neighbor_info['site_index']
            neighbor_nn_info = vnn_loose.get_nn_info(struct, n=neighbor_index)
            neighbor_coord = __get_coordination_string(neighbor_nn_info)
            # Prefix the coordination of this neighbor atom with the identity
            # of the neighber, e.g. 'Cu:Cu-Cu'
            neighbor_element = neighbor_info['site'].species_string
            neighbor_coord_labeled = neighbor_element + ':' + neighbor_coord
            neighborcoord.append(neighbor_coord_labeled)

        # Find the nextnearestcoordination
        nn_info_loose = vnn_loose.get_nn_info(struct, n=uranium_index)
        nextnearestcoordination = __get_coordination_string(nn_info_loose)

        return {'coordination': coordination,
                'neighborcoord': neighborcoord,
                'nextnearestcoordination': nextnearestcoordination}
    # If we get some QHull or ValueError, then just assume that the adsorbate desorbed
    except (QhullError, ValueError):
        return {'coordination': '',
                'neighborcoord': '',
                'nextnearestcoordination': ''}


def remove_adsorbate(adslab):
    '''
    This function removes adsorbates from an adslab and gives you the locations
    of the binding atoms. Note that we assume that the first atom in each adsorbate
    is the binding atom.

    Arg:
        adslab  The `ase.Atoms` object of the adslab. The adsorbate atom(s) must
                be tagged with non-zero integers, while the slab atoms must be
                tagged with zeroes. We assume that for each adsorbate, the first
                atom (i.e., the atom with the lowest index) is the binding atom.
    Returns:
        slab                The `ase.Atoms` object of the bare slab.
        binding_positions   A dictionary whose keys are the tags of the
                            adsorbates and whose values are the cartesian
                            coordinates of the binding site.
    '''
    # Operate on a local copy so we don't propagate changes to the original
    slab = adslab.copy()

    # Remove all the constraints and then re-constrain the slab. We do this
    # because ase does not like it when we delete atoms with constraints.
    slab.set_constraint()
    slab = constrain_slab(slab)

    # Delete atoms in reverse order to preserve correct indexing
    binding_positions = {}
    for i, atom in reversed(list(enumerate(slab))):
        if atom.tag != 0:
            binding_positions[atom.tag] = atom.position
            del slab[i]

    return slab, binding_positions


def __get_coordination_string(nn_info):
    '''
    This helper function takes the output of the `VoronoiNN.get_nn_info` method
    and gives you a standardized coordination string.

    Arg:
        nn_info     The output of the
                    `pymatgen.analysis.local_env.VoronoiNN.get_nn_info` method.
    Returns:
        coordination    A string indicating the coordination of the site
                        you fed implicitly through the argument, e.g., 'Cu-Cu-Cu'
    '''
    coordinated_atoms = [neighbor_info['site'].species_string
                         for neighbor_info in nn_info
                         if neighbor_info['site'].species_string != 'U']
    coordination = '-'.join(sorted(coordinated_atoms))
    return coordination


def calculate_unit_slab_height(atoms, miller_indices, slab_generator_settings=None):
    '''
    Calculates the height of the smallest unit slab from a given bulk and
    Miller cut

    Args:
        atoms                   An `ase.Atoms` object of the bulk you want to
                                make a surface out of
        miller_indices          A 3-tuple of integers representing the Miller
                                indices of the surface you want to make
        slab_generator_settings A dictionary that can be passed as kwargs to
                                instantiate the
                                `pymatgen.core.surface.SlabGenerator` class.
                                Defaults to the settings in
                                `gaspy.defaults.slab_settings`.
    Returns:
        height  A float corresponding the height (in Angstroms) of the smallest
                unit slab
    '''
    if slab_generator_settings is None:
        slab_generator_settings = slab_settings()['slab_generator_settings']
        # We don't care about these things
        del slab_generator_settings['min_vacuum_size']
        del slab_generator_settings['min_slab_size']

    # Instantiate a pymatgen `SlabGenerator`
    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    structure = sga.get_conventional_standard_structure()
    gen = SlabGenerator(initial_structure=structure,
                        miller_index=miller_indices,
                        min_vacuum_size=0.,
                        min_slab_size=0.,
                        **slab_generator_settings)

    # Get and return the height
    height = gen._proj_height
    return height


def find_max_movement(atoms_initial, atoms_final):
    '''
    Given ase.Atoms objects, find the furthest distance that any single atom in
    a set of atoms traveled (in Angstroms)

    Args:
        initial_atoms   `ase.Atoms` of the structure in its initial state
        final_atoms     `ase.Atoms` of the structure in its final state
    Returns:
        max_movement    A float indicating the further movement of any single atom
                        before and after relaxation (in Angstroms)
    '''
    # Calculate the distances for each atom
    distances = atoms_final.positions - atoms_initial.positions

    # Reduce the distances in case atoms wrapped around (the minimum image
    # convention)
    _, movements = find_mic(distances, atoms_final.cell, atoms_final.pbc)
    max_movement = max(movements)

    return max_movement
