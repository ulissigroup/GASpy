import sys
sys.path.append('../../')
from gaspy.utils import get_adsorption_db

get_adsorption_db().db.adsorption.remove({})
