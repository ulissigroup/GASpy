import sys
from gaspy.gasdb import get_adsorption_client

get_adsorption_client().db.adsorption.remove({})
