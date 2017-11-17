'''
This script creates and caches images from our catalog of adsorption sites. We use this as part
of our GASdb interface.
'''

__author__ = 'Zachary W. Ulissi'
__email__ = 'zulissi@andrew.cmu.edu'


from gaspy.utils import get_catalog_db, constrain_slab, get_adsorption_db
from gaspy.defaults import adsorbates_dict
from vasp.mongo import mongo_doc_atoms
from ase.db.plot import atoms2png
from ase.io.png import write_png
from multiprocessing import Pool
import glob
import itertools
from itertools import islice
import pickle
import numpy as np

catalog = get_catalog_db().find()
ads_dict = adsorbates_dict()
del ads_dict['']
del ads_dict['U']
ads_to_run = ads_dict.keys()
ads_to_run = ['CO', 'H']
dump_dir = '/global/cscratch1/sd/zulissi/GASpy_DB/images/'


def writeImages(input):
    doc, adsorbate = input
    atoms = mongo_doc_atoms(doc)
    slab = atoms.copy()
    ads_pos = slab[0].position
    del slab[0]
    ads = ads_dict[adsorbate].copy()
    ads.set_constraint()
    ads.translate(ads_pos)
    adslab = ads + slab
    adslab.cell = slab.cell
    adslab.pbc = [True, True, True]
    adslab.set_constraint()
    adslab = constrain_slab(adslab)
    size = adslab.positions.ptp(0)
    i = size.argmin()
    rotation = ['-90y', '90x', ''][i]
    rotation = ''
    size[i] = 0.0
    scale = min(25, 100 / max(1, size.max()))
    write_png(dump_dir + str(doc['_id']) + '-' + adsorbate + '.png',
              adslab, show_unit_cell=1, scale=scale)
    write_png(dump_dir + str(doc['_id']) + '-' + adsorbate + '-side.png',
              adslab, show_unit_cell=1, rotation='90y, 90z', scale=scale)


def writeAdsorptionImages(doc):
    atoms = mongo_doc_atoms(doc)
    adslab = atoms.copy()
    size = adslab.positions.ptp(0)
    i = size.argmin()
    rotation = ['-90y', '90x', ''][i]
    rotation = ''
    size[i] = 0.0
    scale = min(25, 100 / max(1, size.max()))
    write_png(dump_dir + str(doc['_id']) + '.png', adslab, show_unit_cell=1, scale=scale)
    write_png(dump_dir + str(doc['_id']) + '-side.png', adslab, show_unit_cell=1,
              rotation='90y, 90z', scale=scale)


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size-1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunkdef chunks(iterable, size=10):
    iterator = iter(iterable)


if len(glob.glob('completed_images.pkl')) > 0:
    completed_images = pickle.load(open('completed_images.pkl'))
else:
    completed_images = []


def MakeImages(todo, collection, completed_images):
    pool = Pool(32)
    k = 0
    for chunk in chunks(todo, 10000):
        ids, adsorbates = zip(*chunk)
        uniques, inverse = np.unique(ids, return_inverse=True)
        docs = np.array([collection.find_one({"_id": id}) for id in uniques])
        to_run = zip(docs[inverse], adsorbates)
        pool.map(writeImages, to_run)
        k += 1
        print('%d/%d' % (k*len(to_run), len(todo)))
        completed_images += zip(ids, adsorbates)
        pickle.dump(completed_images, open('completed_images.pkl', 'w'))
    pool.close()


def MakeImagesAdsorption(todo, collection, completed_images):
    pool = Pool(32)
    k = 0
    for chunk in chunks(todo, 10000):
        ids = list(chunk)
        uniques, inverse = np.unique(ids, return_inverse=True)
        docs = np.array([collection.find_one({"_id": id}) for id in uniques])
        to_run = docs[inverse]
        pool.map(writeAdsorptionImages, to_run)
        k += 1
        print('%d/%d' % (k*len(to_run), len(todo)))
        completed_images += ids
        pickle.dump(completed_images, open('completed_images.pkl', 'w'))
    pool.close()


adsorption_ids = get_adsorption_db().db.adsorption.distinct('_id')
unique_combinations = list(itertools.product(adsorption_ids, ads_to_run))
todo = list(set(adsorption_ids) - set(completed_images))
MakeImagesAdsorption(todo, get_adsorption_db().db.adsorption, completed_images)
