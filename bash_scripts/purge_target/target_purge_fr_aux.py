import sys
sys.path.append('../../')
from gaspy import utils

aux_db = utils.get_aux_db()

aux_db.db.atoms.remove({'fwid': int(sys.argv[1])})
aux_db.db.adsorption.remove({'fwid': int(sys.argv[1])})
