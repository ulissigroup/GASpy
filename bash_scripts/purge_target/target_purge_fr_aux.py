import sys
sys.path.append('../../')
from gaspy.utils import get_aux_db

get_aux_db().db.atoms.remove({'fwid': int(sys.argv[1])})
