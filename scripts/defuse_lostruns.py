''' This script will find rockets that are "lost" and then defuse them '''

from gaspy.fireworks_helper_scripts import get_launchpad


# Find the lost runs
lp = get_launchpad()
lost_launch_ids, lost_fw_ids, inconsistent_fw_ids = lp.detect_lostruns()

# We reverse the list, because early-on in the list there's some really bad
# launches that cause this script to hang up. If we run in reverse, then
# it should be able to get the recent ones.
# TODO:  Find a better solution
lost_fw_ids.reverse()

# Defuse them
for _id in lost_fw_ids:
    lp.defuse_fw(_id)
