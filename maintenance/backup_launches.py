'''
This is a script specifically for the Ulissi group to run on NERSC. It
downloads all of the launch directories, tars them, and then stores them in a
designated scratch folder.

Since we run inside Docker containers, we will need to forcefully tell `ssh`
which config file to use. We also need to make sure we are mounting our
`~/.ssh` folder correctly, which should be done by `../open_container*.sh`.
This workflow will probably require you to create a `~/.ssh/config` with
contents similar to this:

Host sm1
    Port 22
    Hostname sm1.cheme.cmu.edu
Host arjuna
        Port 22
        User zulissi
        ProxyCommand ssh -F /global/homes/z/zulissi/.ssh/config -q sm1 nc arjuna.psc.edu 22
Host gilgamesh
        Port 22
        User zulissi
        ProxyCommand ssh -F /global/homes/z/zulissi/.ssh/config -q sm1 nc gilgamesh.cheme.cmu.edu 22
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import warnings
import glob
import os
import multiprocess
from tqdm import tqdm
from gaspy.utils import read_rc
from gaspy.fireworks_helper_scripts import get_launchpad


def fetch_tar_file(launch_id, verify_tars=True):
    '''
    This function will figure out where a Launch ID ran, download the launch
    directory, zip it up, and put it into the backup directory.

    Args:
        launch_id   The FireWorks Launch ID of the folder you want to download
        verify_tars A Boolean indicating whether you want to double-check that
                    any tar folders that are already there are not empty.
    '''
    # We'll put the tar file inside the FireWorks backup directory. The name
    # will be the FireWorks launch ID.
    backup_directory = read_rc('fireworks_info.backup_directory')
    backup_loc = backup_directory + '/%d.tar.gz' % launch_id

    # If the tar ball is empty, delete it so that we can re-fetch correctly
    if len(glob.glob(backup_loc)) > 0 and verify_tars:
        output = os.system('tar -tzf %s >/dev/null' % backup_loc)
        if output != 0:
            os.remove(backup_loc)

    # Only fetch the folder if the tar ball is not already there
    if len(glob.glob(backup_loc)) == 0:

        # Fetch the launch object, which has important information
        try:
            launch = lpad.get_launch_by_id(launch_id)
        except:  # noqa:  E722
            warnings.warn('Could not find Launch ID %i' % launch_id, RuntimeWarning)
            return

        # If the launch isn't finished, then it's not wortch fetching
        if launch.state != 'COMPLETED':
            return

        # Initialize some things before downloading
        cluster = launch.fworker.name
        launch_dir = launch.launch_dir
        # We need to forcefully point to the correct ssh config files, which
        # should be mounted in the GASpy container.
        _download_command = ("rsync -e 'ssh -i /home/.ssh/id_rsa "
                             "-F /home/.ssh/config' -azqP --max-size=100M ")

        # Define the commands to download from each of the clusters
        if 'gilgamesh' in cluster:
            download_command = _download_command + 'gilgamesh:%s /tmp/%s' % (launch.launch_dir, launch_id)
        elif 'arjuna' in cluster:
            download_command = _download_command + 'arjuna:%s /tmp/%s' % (launch.launch_dir, launch_id)
        elif 'Cori' in cluster or '/global/project/projectdirs/m2755/' in launch_dir:
            download_command = _download_command + '%s /tmp/%s' % (launch_dir, launch_id)
            # Some of our old fireworks are in weird locations. We handle those
            # here.
            if '/global/u2/z/zulissi' in launch_dir:
                launch_dir = '/global/project/projectdirs/m2755/fireworks_zu/fireworks/' + launch_dir[30:]
        # Just in case something weird happens
        else:
            warnings.warn('Unrecognized cluster (%s) for FWID %i' % (cluster, launch.fw_id),
                          RuntimeWarning)
            download_command = (':')
            return

        # Download the folder into `/tmp`, tar it, move it, and delete the
        # original download
        os.system(download_command)
        launch_dir_folder = launch.launch_dir.split('/')[-1]
        os.system("(cd /tmp/%s/%s && tar -czf %s *)" % (launch_id, launch_dir_folder, backup_loc))
        os.system("rm -r /tmp/%s" % launch_id)


# Get all of the launch IDs of successful runs
lpad = get_launchpad()
fwids = lpad.get_fw_ids({"state": "COMPLETED"})
launch_ids = []
for fwid in tqdm(fwids):
    fw = lpad.get_fw_by_id(fwid)
    for launch in fw.launches:
        launch_ids.append(launch.launch_id)

# Multiprocess downloading
with multiprocess.Pool(68) as pool:
    iterator = pool.imap(fetch_tar_file, launch_ids, chunksize=20)
    list(tqdm(iterator, total=len(launch_ids)))
