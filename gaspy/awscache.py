''' Function to cache png pictures of atoms objects to aws s3 buckets '''

__authors__ = [ 'Zack Ulissi']
__emails__ = [ 'zulissi@andrew.cmu.edu']

from tqdm import tqdm
from gaspy.utils import read_rc
from gaspy.atoms_operators import tile_atoms
from ase.io.png import write_png
from gaspy.mongo import make_atoms_from_doc
from multiprocess import Pool
import boto3
from botocore.exceptions import ClientError

def CacheImages(doc):
    '''
    Make image of atoms object and pushes it to AWS.

    Args:
        doc    gaspy document. Must contain the $atoms and $results sections
                that make this a valid mongo-ase document
    '''
    image_name = str(doc['mongo_id']) + '.png'
    side_image_name = str(doc['mongo_id']) + '-side.png'
    
    try:
        s3_client.head_object(Bucket='catalyst-thumbnails', Key=image_name)
    except ClientError:
        atoms = make_atoms_from_doc(doc)
        atoms.set_constraint()
        adslab,repeats = tile_atoms(atoms, 10,10)
        size = adslab.positions.ptp(0)
        i = size.argmin()
        rotation = ['-90y', '90x', ''][i]
        rotation = ''
        size[i] = 0.0
        scale = min(25, 100 / max(1, size.max()))
        write_png(read_rc()['temp_directory'] + image_name, adslab, show_unit_cell=1, scale=scale)
        write_png(read_rc()['temp_directory'] + side_image_name, adslab, show_unit_cell=1,
                  rotation='90y, 90z', scale=scale)
        upload_file(read_rc()['temp_directory'] + image_name,'catalyst-thumbnails', image_name)
        upload_file(read_rc()['temp_directory'] + side_image_name,'catalyst-thumbnails', side_image_name)
        
def upload_file(file_name, bucket, object_name):
    """Upload a file to an S3 bucket, taken from AWS examples

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

s3_client = boto3.client('s3', aws_access_key_id=read_rc()['aws_login_info']['aws_access_key_id'], 
                         aws_secret_access_key=read_rc()['aws_login_info']['aws_secret_access_key'])

def cache_docs_to_images(docs, n_procs = 1):
    '''
    Make image of atoms object and pushes it to AWS.

    Args:
        docs    list of gaspy documents. Must contain the $atoms 
                and $results sections that make this a valid mongo-ase document
        n_procs number of cores to use. 1 uses single-thread map, more than 
                1 uses multiprocess to multithread it. 
    '''
    if n_procs == 1:
        list(tqdm(map(CacheImages, docs)))
    else:
        with Pool(n_procs) as pool:
            list(tqdm(pool.imap(CacheImages, docs, chunksize=10)))
    
