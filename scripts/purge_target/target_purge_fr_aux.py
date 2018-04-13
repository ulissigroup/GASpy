import sys
from gaspy.gasdb import get_atoms_client

client = get_atoms_client()

client.db.atoms.remove({'fwid': int(sys.argv[1])})
client.db.adsorption.remove({'processed_data.FW_info.slab+adsorbate': int(sys.argv[1])})
