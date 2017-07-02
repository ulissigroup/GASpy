from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder


def fingerprint(atoms, siteind):
    '''
    This function is used to fingerprint an atoms object, where the "fingerprint" is a dictionary
    of properties that we believe may be adsorption motifs.
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
