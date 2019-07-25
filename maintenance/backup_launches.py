import glob
import os
import multiprocess
from tqdm import tqdm
from gaspy.utils import read_rc
from gaspy.fireworks_helper_scripts import get_launchpad


def fetch_tar_file(launch_id, verify_tars=True):
    backup_directory = read_rc()['launches_backup_directory']
    backup_loc = backup_directory + '/%d.tar.gz' % launch_id
    if len(glob.glob(backup_loc)) > 0 and verify_tars:
        output = os.system('tar -tzf %s >/dev/null' % backup_loc)
        if output != 0:
            os.remove(backup_loc)
    if len(glob.glob(backup_loc)) == 0:
        try:
            launch = lpad.get_launch_by_id(launch_id)
        except:
            print('could not find launch id %d' % launch_id)
            return

        if launch.state != 'COMPLETED':
            print(launch.fw_id)
            return

        cluster = launch.fworker.name
        if cluster == 'gilgamesh':
            os.system("rsync -azqP --max-size=100M gilgamesh:%s /tmp/%s" % (launch.launch_dir, launch_id))
        elif 'arjuna' in cluster:
            os.system("rsync -azqP --max-size=100M arjuna:%s /tmp/%s" % (launch.launch_dir, launch_id))
        elif 'Cori' in cluster or '/global/project/projectdirs/m2755/' in launch.launch_dir:
            launch_dir = launch.launch_dir
            if '/global/u2/z/zulissi' in launch_dir:
                launch_dir = '/global/project/projectdirs/m2755/fireworks_zu/fireworks/' + launch_dir[30:]

            os.system("rsync -azqP --max-size=100M %s /tmp/%s" % (launch_dir, launch_id))
        else:
            print('unknown cluster!: %s' % cluster)
            print(launch.fw_id)
        launch_dir_folder = launch.launch_dir.split('/')[-1]
        os.system("(cd /tmp/%s/%s && tar -czf %s *)" % (launch_id, launch_dir_folder, backup_loc))
        os.system("rm -r /tmp/%s" % launch_id)


lpad = get_launchpad()

all_fw_ids = lpad.get_fw_ids({"state": "COMPLETED"})

launch_list = []
for fwid in tqdm(all_fw_ids, decs='Finding launch IDs', unit='fw'):
    fw = lpad.get_fw_by_id(fwid)
    for launch in fw.launches:
        launch_list.append(launch.launch_id)


with multiprocess.Pool(4) as pool:
    iterator = pool.imap(fetch_tar_file, launch_list, chunksize=20)
    list(tqdm(iterator, desc='Getting tar files', total=len(launch_list)), unit='fw')
