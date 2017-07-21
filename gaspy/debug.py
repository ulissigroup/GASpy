'''
This module is not meant to be called by GASpy, but rather by a user trying to
investigate/debug things going on with GASpy.
'''
import sys
import pdb
from pprint import pprint
from vasp.mongo import mongo_doc_atoms
from . import utils
from . import defaults
from . import fireworks_helper_scripts as fwhs


class AuxPull(object):
    '''
    Given a set of kwargs, this class will be able to filter out pertinent information from
    the Aux DB and then return it to the user in different forms (depending on the method
    used).

    Note that a recurring theme in these methods is to return a list of objects if the
    filter returns many matches, or return a single object in the filter returns only
    one match.
    '''
    def __init__(self, **kwargs):
        '''
        Get our launchpad, the mongo docs created by vaspy, and the number of matches.

        Input:
            kwargs  Each argument name should be a key that AuxDB can use to find an entry,
                    and each argument value should be the search string the user wants
                    to filter by
        '''
        self.lpad = fwhs.get_launchpad()
        self.docs = utils.get_aux_db().find(kwargs)
        self.n = docs.count()

        # If there are no matches, tell the user.
        if not self.n:
            print('DBPull fail:  no matching rows in Aux DB')

    def mdocs(self):
        ''' Return the mongo document(s) for the matching entries '''
        if self.n == 1:
            return self.docs[0]
        elif self.n > 1:
            return self.docs

    def atoms(self):
        ''' Return the ase.Atoms object(s) for the matching entries '''
        if self.n == 1:
            return mongo_doc_atoms(self.docs[0])
        elif self.n > 1:
            return [mongo_doc_atoms(doc) for doc in self.docs]

    def fw(self):
        ''' Return the fw object(s) for the matching entries '''
        if self.n == 1:
            return self.lpad.get_fw_by_id(self.docs[0]['fwid'])
        elif self.n > 1:
            return [self.lpad.get_fw_by_id(doc['fwid']) for doc in self.docs]
