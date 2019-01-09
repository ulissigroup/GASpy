'''
This submodule contains various functions that operate on both `ase.Atoms`
objects and `pymatgen.Structure` objectsto do various things.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import math
import numpy as np
from ase.build import rotate
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from .utils import unfreeze_dict


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
