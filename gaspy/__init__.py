'''
This package creates various classes, funcions, and tasks to automatically perform
DFT simulations and database the results
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


from . import utils
from . import defaults
from . import gasdb
from . import fireworks_helper_scripts
from . import vasp_functions
from . import debug

# Luigi cannot handle modules that have relative imports, which means that
# task-containing modules cannot be part of packages. Do not try to add them
# to __init__.py, because that will effectively make that module import itself,
# which creates redundancy of tasks when using Luigi.
# from . import tasks
