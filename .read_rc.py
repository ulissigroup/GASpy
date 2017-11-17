'''
This script reads the .gaspyrc.json file and prints out the config that you ask for.

Input:
    [1]     A string for the key that you want printed
Output:
    stdout  The value of the key that you input, as per the .gaspyrc.json
'''

import sys
from gaspy.utils import read_rc


# Pull out the input
key = sys.argv[1]

# Read the .gaspyrc.json file
configs = read_rc()

# Print it out
for config, value in configs.iteritems():
    if config == key:
        print(value)
        break
