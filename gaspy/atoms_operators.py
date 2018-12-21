'''
This submodule contains various functions that operate on both `ase.Atoms`
objects and `pymatgen.Structure` objectsto do various things.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import numpy as np
from ase.build import rotate
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from .mongo import make_doc_from_atoms


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


def make_slab_docs_from_structs(slab_structures):
    '''
    This function will take a list of pymatgen.Structure slabs, convert them
    into `ase.Atoms` objects, orient the slabs upwards, fix the subsurface
    atoms, and then turn those atoms objects into dictionaries (i.e.,
    documents). This function will also enumerate and return new documents for
    invertible slabs that you give it, so the number of documents you get out
    may be greater than the number of structures you put in.

    Arg:
        slab_structures     A list of pymatgen.Structure objects. They should
                            probably be created by the
                            `make_slabs_from_bulk_atoms` function, but you do
                            you.
    Returns:
        docs    A list of dictionaries (also known as "documents", because
                they'll eventually be put into Mongo as documents) that contain
                information about slabs. These documents can be fed to the
                `gaspy.mongo.make_atoms_from_docs` function to be turned
                into `ase.Atoms` objects. These documents also contain
                the 'shift' and 'top' fields to indicate the shift/termination
                of the slab and whether or not the slab is oriented upwards
                with respect to the way it was enumerated originally by
                pymatgen.
    '''
    docs = []
    for struct in slab_structures:
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms = orient_atoms_upwards(atoms)

        # Convert each slab into dictionaries/documents
        atoms_constrained = constrain_slab(atoms)
        doc = make_doc_from_atoms(atoms_constrained)
        doc['shift'] = struct.shift
        doc['top'] = True
        docs.append(doc)

        # If slabs are invertible (i.e., are not symmetric about the x-y
        # plane), then flip it and make another document out of it.
        if is_structure_invertible(struct) is True:
            atoms_flipped = flip_atoms(atoms)
            atoms_flipped_constrained = constrain_slab(atoms_flipped)
            doc_flipped = make_doc_from_atoms(atoms_flipped_constrained)
            doc_flipped['shift'] = struct.shift
            doc_flipped['top'] = False
            docs.append(doc_flipped)

    return docs


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


#def remove_adsorbate(adslab):
#    '''
#    This function removes adsorbates from an adslab, as long as you've tagged
#    the adslab correctly.
#
#    Input:
#        adslab  The `ase.Atoms` object of the adslab. The adsorbate atom(s) must
#                be tagged with non-zero integers, while the slab atoms must be
#                tagged with zeroes.
#    Outputs:
#        slab    The `ase.Atoms` object of the bare slab.
#    '''
#    # Work on a copy so that we don't modify the original
#    slab = adslab.copy()
#
#    # We remove the atoms in reverse order so that the indices of the atoms
#    # don't change while we're iterating through them.
#    for i, atom in reversed(list(enumerate(slab))):
#        if atom.tag != 0:
#            del slab[i]
#    return slab
