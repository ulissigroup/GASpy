# This script is a simple script to write a hex string representation of a traj
# file to a file so that it can be passed as input to vasp


def atoms_hex_to_file(fname_out, atomHex):
    # Dump the hex encoded string to a local file
    with open(fname_out, 'w') as fhandle:
        fhandle.write(atomHex.decode('hex'))
    #print(read(fname_out))
