'''
This is an example script is used to make FireWorks rockets---i.e., submit
calculations for---all of the adsorption sites on a given set of surfaces.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks import run_tasks
from gaspy.tasks.submit_calculations.adsorption_calculations import AllSitesOnSurfaces


adsorbates = [['CO'], ['H']]
mpids = ['mp-2', 'mp-30']
millers = [[1, 1, 1], [1, 0, 0]]
max_rockets = 20

rocket_builder = AllSitesOnSurfaces(adsorbates_list=adsorbates,
                                    mpid_list=mpids,
                                    miller_list=millers,
                                    max_rockets=max_rockets)

tasks = [rocket_builder]
run_tasks(tasks)
