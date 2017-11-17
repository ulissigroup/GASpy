import sys
from gaspy.gasdb import get_atoms_client

get_atoms_client().db.atoms.remove({})
