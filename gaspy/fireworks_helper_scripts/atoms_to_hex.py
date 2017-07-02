def atoms_to_hex(atoms):
    atoms.write('temp.traj')
    return open('temp.traj').read().encode('hex')
