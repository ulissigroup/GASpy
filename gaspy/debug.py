'''
This module is not meant to be called by GASpy, but rather by a user trying to
investigate/debug things going on with GASpy.
'''
import pdb  # noqa: F401
from vasp.mongo import mongo_doc_atoms
from ase.visualize import view
from . import utils
from . import fireworks_helper_scripts as fwhs


class AuxPull(object):
    '''
    Given a set of kwargs, this class will be able to filter out pertinent information from
    the Aux DB and then return it to the user in different forms (depending on the method
    used).

    Note that a recurring theme in these methods is to have pairs of methods to return singular
    objects and multiple objects (e.g., mdocs and mdoc). The plural ones return lists, while
    the singular ones return single objects.
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
        self.n = self.docs.count()

        # If there are no matches, tell the user.
        if not self.n:
            print('DBPull fail:  no matching rows in Aux DB')

    def mdoc(self):
        ''' Return the mongo document for the first matching entry '''
        return self.docs[0]

    def mdocs(self):
        ''' Return the mongo documents for all matching entries '''
        return self.docs

    def atoms(self):
        ''' Return the atoms object for the first matching entry '''
        return mongo_doc_atoms(self.docs[0])

    def atomss(self):
        ''' Return the atoms objects for all matching entries '''
        return [mongo_doc_atoms(doc) for doc in self.docs]

    def fw(self):
        ''' Return the fw object for the first matching entry '''
        return self.lpad.get_fw_by_id(self.docs[0]['fwid'])

    def fws(self):
        ''' Return the fw objects for all matching entries '''
        return [self.lpad.get_fw_by_id(doc['fwid']) for doc in self.docs]


def check_atoms(fwids):
    '''
    Given a Fireworks ID number, this function will print out the energy and forces of
    the system and then pull up the ase visualizer for you to look at it. If you supply
    it with a list, then it will open all of them.
    '''
    # EAFP to open the list or a single value
    try:
        for fwid in fwids:
            atom_puller = AuxPull(fwid=fwid)
            atoms = atom_puller.atoms()
            print('Looking at %s' % atoms.get_chemical_formula())
            print('Energy:  %s eV' % atoms.get_potential_energy())
            print('Forces:\n%s' % atoms.get_forces())
            view(atoms)
    except TypeError:
        atom_puller = AuxPull(fwid=fwids)
        atoms = atom_puller.atoms()
        print('Looking at %s' % atoms.get_chemical_formula())
        print('Energy:  %s eV' % atoms.get_potential_energy())
        print('Forces:\n%s' % atoms.get_forces())
        view(atoms)
