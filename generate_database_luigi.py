"""
This module houses various functions and classes that Luigi uses to set up calculations that can
be submitted to Fireworks. This is intended to be used in conjunction with a submission file, an
example of which is named "adsorbtionTargets.py".
"""


from collections import OrderedDict
import copy
import math
from math import ceil
import glob
import cPickle as pickle
import numpy as np
from numpy.linalg import norm
import pymatgen
from pymatgen.matproj.rest import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder
from ase.db import connect
from ase import Atoms
import ase.io
from ase.utils.geometry import rotate
from ase.constraints import *
from ase.calculators.singlepoint import SinglePointCalculator
from fireworks import LaunchPad, Firework, Workflow, PyTask
from fireworks_helper_scripts import atoms_hex_to_file, atoms_to_hex
from findAdsorptionSites import find_adsorption_sites
from vasp_settings_to_str import vasp_settings_to_str
from vasp.mongo import MongoDatabase, mongo_doc, mongo_doc_atoms
import luigi


def get_launchpad():
    return LaunchPad(host='gilgamesh.cheme.cmu.edu',
                     name='fw_zu_vaspsurfaces',
                     username='admin_zu_vaspsurfaces',
                     password='$TPAHPmj',
                     port=30000)


def adsorbate_dictionary(adsorbate):
    """
    This is a helper function to take an adsorbate as a string (e.g. 'CO') and attempt to
    return an atoms object for it, primarily as a way to count the number of constitutent
    atoms in the adsorbate
    """
    # First, try to create an [atoms class] from the input
    try:
        atoms = Atoms(adsorbate)
    # If that doesn't work, then look for the adsorbate in the "atomDict" object
    except:
        try:
            atoms = atomDict[adsorbate]
        # If that doesn't work, then alert the user and move on
        except:
            print('Error: Adsorbate name may not be in the "atomDict" dictionary')
    # Return the number of constraints added
    return atoms


def set_constraints(atoms, numSlabAtoms, zcutoff=3.):
    """
    Define a function, "setConstraints" to impose slab constraints prior to relaxation.
    Inputs
    atoms     ASE-atoms class of the slab to be constrained
    zcutoff   The threshold to see if other atoms are in the same plane as the highest atom
    """

    # Initialize
    constraints = []            # This list will contain the various constraints we will impose
    atomscopy = atoms.copy()    # Create a local copy of the atoms class to work on

    scaled_positions = atomscopy.get_scaled_positions()
    # Constrain atoms except for the top layer
    # Find height of the highest atom, and then define it as the "maxSlabZPos" float.
    maxSlabZPos = np.max([pos[2] for pos in scaled_positions[0:numSlabAtoms]])
    minSlabZPos = np.min([pos[2] for pos in scaled_positions[0:numSlabAtoms]])

    # Add the constraint, which is a binary list (i.e., 1's & 0's) used to identify which atoms
    # to fix or not. The indices of the list correspond to the indices of the atoms in the "atoms".
    if atoms.cell[2, 2] > 0:
        constraints.append(FixAtoms(mask=[pos[2] < maxSlabZPos-(zcutoff/norm(atomscopy.cell[2]))
                                          for pos in scaled_positions]))
    else:
        constraints.append(FixAtoms(mask=[pos[2] > minSlabZPos+(zcutoff/norm(atomscopy.cell[2]))
                                          for pos in scaled_positions]))

    # Enact the constraints on the local atoms class
    atomscopy.set_constraint(constraints)
    return atomscopy


def make_firework(atomin, namein, vaspin, threshold=40, maxMiller=3):
    """
    This function makes a simple vasp relaxation firework
    atomin: atoms object to relax
    namein: dictionary of tags/etc to use at the fireworks name
    vaspin: dictionary of vasp settings to pass to Vasp()
    threshold: max number of atoms to submit, mainly as a way to prevent
        overly-large simulations from getting run
    maxMiller: maximum miller index to submit, so that be default miller indices
        above 3 won't get submitted by accident
    """
    if len(atomin) > threshold:
        print('too many atoms! '+str(namein))
        return
    if 'miller' in namein and np.max(eval(str(namein['miller']))) > maxMiller:
        print('too high miller! '+str(namein))
        return
    # Generate a string representation that we can pass to the job as input
    atom_hex = atoms_to_hex(atomin)
    # Two steps - write the input structure to an input file, then relax that traj file
    write_surface = PyTask(func='fireworks_helper_scripts.atomsHexToFile',
                           args=['slab_in.traj',
                                 atom_hex]
                          )
    opt_bulk = PyTask(func='vasp_scripts.runVasp',
                      args=['slab_in.traj',
                            'slab_relaxed.traj',
                            vaspin],
                      stored_data_varname='opt_results')

    # Package the tasks into a firework, the fireworks into a workflow,
    # and submit the workflow to the launchpad
    firework = Firework([write_surface, opt_bulk], name=namein)
    return firework


def running_fireworks(name_dict, launchpad):
    """
    Return the running, ready, or completed fireworks on the launchpad with a given name
    namedict: name dictionary to search for
    launchpad: launchpad to use
    """
    # Make a mongo query
    name = {}
    # Turn a nested dictionary into a series of mongo queries
    for key in name_dict:
        if isinstance(name_dict[key], dict) or isinstance(name_dict[key], OrderedDict):
            for key2 in name_dict[key]:
                name['name.%s.%s'%(key, key2)] = name_dict[key][key2]
        else:
            if key == 'shift':
                # Search for a range of shift parameters up to 4 decimal place
                shift = float(np.round(name_dict[key], 4))
                name['name.%s'%key] = {'$gte':shift-1e-4, '$lte':shift+1e-4}
            else:
                name['name.%s'%key] = name_dict[key]

    # Get all of the fireworks that are completed, running, or ready (i.e., not fizzled
    # or defused.
    fw_ids = launchpad.get_fw_ids(name)
    fw_list = []
    for fwid in fw_ids:
        fw = launchpad.get_fw_by_id(fwid)
        if fw.state in ['RUNNING', 'COMPLETED', 'READY']:
            fw_list.append(fwid)
    # Return the matching fireworks
    return fw_list


def get_db():
    """ Get a handle to the results database """
    return MongoDatabase(host='ds117909.mlab.com',
                         port=17909,
                         user='ulissi_admin',
                         password='ulissi_admin',
                         database='ulissigroup_test',
                         collection='atoms')


class UpdateDB(luigi.Task):
    """
    This is a task that looks at the fireworks database and loads the values into the
    results database (the one in getDB())
    """
    def run(self):
        launchpad = LaunchPad(host='gilgamesh.cheme.cmu.edu',
                              name='fw_zu_vaspsurfaces',
                              username='admin_zu_vaspsurfaces',
                              password='$TPAHPmj',
                              port=30000)

        # Create a class, "con", that has methods to interact with the database.
        with get_db() as MD:

            # A list of integers containing the Fireworks job ID numbers that have been
            # added to the database already
            all_inserted_fireworks = [a['fwid'] for a in MD.find({'fwid':{'$exists':True}})]

            # Given a Fireworks ID, this function will return the "atoms" [class] and
            # "vasp_settings" [str] used to perform the relaxation
            def getFireworkInfo(fw):
                atomshex = fw.launches[-1].action.stored_data['opt_results'][1]
                atoms_hex_to_file('atom_temp.traj', atomshex)
                atoms = ase.io.read('atom_temp.traj')
                starting_atoms = ase.io.read('atom_temp.traj', index=0)
                vasp_settings = fw.name['vasp_settings']
                vasp_settings = vasp_settings_to_str(vasp_settings)
                return atoms, starting_atoms, atomshex, vasp_settings

            # Get all of the completed fireworks
            optimization_fws = launchpad.get_fw_ids({'state':'COMPLETED'})

            # For each fireworks object, turn the results into a mongo doc and submit
            # into the database
            for fwid in optimization_fws:
                if fwid not in all_inserted_fireworks:
                    fw = launchpad.get_fw_by_id(fwid)
                    # Get the information from the class we just pulled from the launchpad
                    atoms, starting_atoms, trajectory, vasp_settings = getFireworkInfo(fw)
                    doc = mongo_doc(atoms)
                    doc['initial_configuration'] = mongo_doc(starting_atoms)
                    doc['fwname'] = fw.name
                    if 'miller' in fw.name:
                        if isinstance(fw.name['miller'], str) or isinstance(fw.name['miller'], unicode):
                            doc['fwname']['miller'] = eval(doc['fwname']['miller'])
                    doc['fwid'] = fwid
                    doc['directory'] = fw.launches[-1].launch_dir
                    if fw.name['calculation_type'] == 'unit cell optimization':
                        doc['type'] = 'bulk'
                    elif fw.name['calculation_type'] == 'gas phase optimization':
                        doc['type'] = 'gas'
                    elif fw.name['calculation_type'] == 'slab optimization':
                        doc['type'] = 'slab'
                        if 'shift' not in doc['fwname']:
                            doc['fwname']['shift'] = 0
                        doc['fwname']['shift'] = float(np.round(doc['fwname']['shift'], 4))
                    elif fw.name['calculation_type'] == 'slab+adsorbate optimization':
                        doc['type'] = 'slab+adsorbate'
                        if 'shift' not in doc['fwname']:
                            doc['fwname']['shift'] = 0
                        doc['fwname']['shift'] = float(np.round(doc['fwname']['shift'], 4))

                    MD.write(doc)

        #Touch the token
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget('./updatedDB.token')


class WriteRow(luigi.Task):
    """
    his class defines a task to get a result from the database of fireworks results
    (in getDB()). If the result is not present, we should yield a dependency that will
    create and relax the necessary result
    """
    # Calctype is one of 'gas','slab','bulk','slab+adsorbate'
    calctype=luigi.Parameter()

    # Parameters is a nested dictionary of parameters
    parameters=luigi.DictParameter()

    def requires(self):
        """
        This function compares a search dictionary with another dictionary row,
        and returns true if all entries in search match the corresponding entry in row
        """
        def logical_fun(row, search):
            rowdict = row.__dict__
            for key in search:
                if key not in rowdict:
                    return False
                elif rowdict[key] != search[key]:
                    return False
            return True

        # Define a dictionary that will be used to search the results database and find
        # the correct entry
        if self.calctype == 'gas':
            search_strings = {'type':'gas', 'fwname.gasname':self.parameters['gas']['gasname']}
            for key in self.parameters['gas']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s'%key] = self.parameters['gas']['vasp_settings'][key]
        elif self.calctype == 'slab':
            search_strings = {'type':'slab',
                              'fwname.miller':
                              list(self.parameters['slab']['miller']),
                              'fwname.top':self.parameters['slab']['top'],
                              'fwname.shift':self.parameters['slab']['shift'],
                              'fwname.mpid':self.parameters['bulk']['mpid']}
            for key in self.parameters['slab']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s'%key] = self.parameters['slab']['vasp_settings'][key]
        elif self.calctype == 'bulk':
            search_strings = {'type':'bulk',
                              'fwname.mpid':self.parameters['bulk']['mpid']}
            for key in self.parameters['bulk']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s'%key] = self.parameters['bulk']['vasp_settings'][key]
        elif self.calctype == 'slab+adsorbate':
            search_strings = {'type':'slab+adsorbate',
                              'fwname.miller':list(self.parameters['slab']['miller']),
                              'fwname.top':self.parameters['slab']['top'],
                              'fwname.shift':self.parameters['slab']['shift'],
                              'fwname.mpid':self.parameters['bulk']['mpid'],
                              'fwname.adsorbate':self.parameters['adsorption']['adsorbates'][0]['name']}
            for key in self.parameters['adsorption']['vasp_settings']:
                if key not in ['nsw']:
                    search_strings['fwname.vasp_settings.%s'%key] = self.parameters['adsorption']['vasp_settings'][key]
            if 'adsorption_site' in self.parameters['adsorption']['adsorbates'][0]:
                search_strings['fwname.adsorption_site'] = self.parameters['adsorption']['adsorbates'][0]['adsorption_site']

        if 'fwname.shift' in search_strings:
            search_strings['fwname.shift'] = np.round(search_strings['fwname.shift'], 4)

        # Grab all of the matching entries in the database
        with get_db() as MD:
            self.matching_row = list(MD.find(search_strings))

        # If there are no matching entries, we need to yield a requirement that will
        # generate the necessary unrelaxed structure
        if len(self.matching_row) == 0:
            if self.calctype == 'slab':
                return GenerateSurfaces(OrderedDict(bulk=self.parameters['bulk'], slab=self.parameters['slab']))
            if self.calctype == 'slab+adsorbate':
                return FingerprintGeneratedStructures(self.parameters)
            if self.calctype == 'bulk':
                return GenerateBulk({'bulk':self.parameters['bulk']})
            if self.calctype == 'gas':
                return GenerateGas({'gas':self.parameters['gas']})



    def run(self):
        # If there are matching entries, this is easy, just dump the matching entries
        # into a pickle file
        if len(self.matching_row) > 0:
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump(self.matching_row, open(self.temp_output_path, 'w'))
        # Otherwise, we're missing a structure, so we need to submit whatever the
        # requirement returned
        else:
            launchpad = get_launchpad()
            tosubmit = []
            if self.calctype == 'bulk':
                name = {'vasp_settings':self.parameters['bulk']['vasp_settings'],
                        'mpid':self.parameters['bulk']['mpid'],
                        'calculation_type':'unit cell optimization'}
                if len(running_fireworks(name, launchpad)) == 0:
                    atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
                    tosubmit.append(make_firework(atoms, name, self.parameters['bulk']['vasp_settings']))
            if self.calctype == 'gas':
                name = {'vasp_settings':self.parameters['gas']['vasp_settings'],
                        'gasname':self.parameters['bulk']['gasname'],
                        'calculation_type':'gas phase optimization'}
                if len(running_fireworks(name, launchpad)) == 0:
                    atoms = ase.io.read(self.input().fn)
                    tosubmit.append(make_firework(atoms, name, self.parameters['gas']['vasp_settings']))
            if self.calctype == 'slab':
                slab_list = pickle.load(self.input().open())
                atomlist = [mongo_doc_atoms(slab) for slab in slab_list
                            if float(np.round(slab['tags']['shift'],4)) == float(np.round(self.parameters['slab']['shift'],4))
                            and slab['tags']['top'] == self.parameters['slab']['top']
                           ]
                if len(atomlist) > 1:
                    print('matching atoms! something is weird')
                    print(self.input().fn)
                else:
                    atoms = atomlist[0]
                name = {'shift':self.parameters['slab']['shift'],
                        'mpid':self.parameters['bulk']['mpid'],
                        'miller':self.parameters['slab']['miller'],
                        'top':self.parameters['slab']['top'],
                        'vasp_settings':self.parameters['slab']['vasp_settings'],
                        'calculation_type':'slab optimization',
                        'num_slab_atoms':len(atoms)}
                if len(running_fireworks(name, launchpad)) == 0:
                    tosubmit.append(make_firework(atoms, name, self.parameters['slab']['vasp_settings']))
            if self.calctype == 'slab+adsorbate':
                fpd_structs = pickle.load(self.input().open())
                def matchFP(entry, fp):
                    for key in fp:
                        if type(entry[key])==list:
                            if sorted(entry[key])!=sorted(fp[key]):
                                return False
                        else:
                            if entry[key] != fp[key]:
                                return False
                    return True
                # If there is an 'fp' key in parameters['adsorption']['adsorbates'][0], we
                # search for a site with the correct fingerprint, otherwise we search for an
                # adsorbate at the correct location
                if 'fp' in self.parameters['adsorption']['adsorbates'][0]:
                    matching_rows = [row for row in fpd_structs
                                     if matchFP(row, self.parameters['adsorption']['adsorbates'][0]['fp'])]
                else:
                    if self.parameters['adsorption']['adsorbates'][0]['name'] != '':
                        matching_rows = [row for row in fpd_structs
                                         if fpd_structs['adsorption_site'] == self.parameters['adsorption']['adsorbates'][0]['adsorption_site']]
                    else:
                        matching_rows = [row for row in fpd_structs]
                if self.parameters['adsorption']['adsorbates'][0]['name'] == '':
                    matching_rows = matching_rows[0:1]
                elif 'numtosubmit' in self.parameters['adsorption']:
                    matching_rows = matching_rows[0:self.parameters['adsorption']['numtosubmit']]

                for row in matching_rows:
                    name = {'mpid':self.parameters['bulk']['mpid'],
                            'miller':self.parameters['slab']['miller'],
                            'top':self.parameters['slab']['top'],
                            'adsorbate':self.parameters['adsorption']['adsorbates'][0]['name'],
                            'adsorption_site':row['adsorption_site'],
                            'vasp_settings':self.parameters['adsorption']['vasp_settings'],
                            'num_slab_atoms':self.parameters['adsorption']['num_slab_atoms'],
                            'slabrepeat':self.parameters['adsorption']['slabrepeat'],
                            'calculation_type':'slab+adsorbate optimization'}

                    if name['adsorbate'] == '':
                        del name['adsorption_site']
                    if len(running_fireworks(name, launchpad)) == 0:
                        atoms = row['atoms']
                        tosubmit.append(make_firework(atoms,
                                                      name,
                                                      self.parameters['adsorption']['vasp_settings']))
            # If we've found a structure that needs submitting, do so
            print('tosubmit: '+str(tosubmit))
            tosubmit = [a for a in tosubmit if a is not None]
            if len(tosubmit) > 0:
                wflow = Workflow(tosubmit, name='vasp optimization')
                launchpad.add_wf(wflow)
                print(wflow)

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class GenerateBulk(luigi.Task):
    parameters = luigi.DictParameter()

    def run(self):
        with MPRester("MGOdX3P4nI18eKvE") as m:
            structure = m.get_structure_by_material_id(self.parameters['bulk']['mpid'])
            # convert the structure class to an ASE atoms class...
            atoms = AseAtomsAdaptor.get_atoms(structure)
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump([mongo_doc(atoms)], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class GenerateSurfaces(luigi.Task):
    parameters = luigi.DictParameter()

    def requires(self):
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed'] == True:
            return GenerateBulk(parameters={'bulk':self.parameters['bulk']})
        else:
            return WriteRow(calctype='bulk', parameters={'bulk':self.parameters['bulk']})

    def run(self):
        atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
        structure = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        structure = sga.get_conventional_standard_structure()
        gen = SlabGenerator(structure,
                            self.parameters['slab']['miller'],
                            **self.parameters['slab']['slab_generate_settings'])
        slabs = gen.get_slabs(**self.parameters['slab']['get_slab_settings'])
        slabsave = []
        for slab in slabs:
            # If there's only one matching slab the shift doesn't matter so we'll just label it '0'
            if len([a for a in slabs if a.miller_index == slab.miller_index]) == 1:
                shift = 0
            else:
                shift = slab.shift

            # Create an atoms class for this particular slab, "atoms_slab"
            atoms_slab = AseAtomsAdaptor.get_atoms(slab)
            # Then reorient the "atoms_slab" class so that the surface of the slab is pointing upwards
            # in the z-direction
            rotate(atoms_slab,
                   atoms_slab.cell[2], (0, 0, 1),
                   atoms_slab.cell[0], [1, 0, 0],
                   rotate_cell=True)

            # Save the slab, but only if it isn't already in the database
            tags = {'type':'slab',
                    'top':True,
                    'mpid':self.parameters['bulk']['mpid'],
                    'miller':self.parameters['slab']['miller'],
                    'shift':shift,
                    'num_slab_atoms':len(atoms_slab),
                    'relaxed':False,
                    'slab_generate_settings':self.parameters['slab']['slab_generate_settings'],
                    'get_slab_settings':self.parameters['slab']['get_slab_settings']}
            slabdoc = mongo_doc(set_constraints(atoms_slab, len(atoms_slab)))
            slabdoc['tags'] = tags

            slabsave.append(slabdoc)

            # If the top of the cut is not identical to the bottom, then save the bottom slab to the
            # database, as well. To do this, we first pull out the sga class of this particular
            # slab, "sga_slab". Again, we use a symmetry finding tolerance of 0.1 to be consistent
            # with MP
            sga_slab = SpacegroupAnalyzer(slab, symprec=0.1)
            # Then use the "sga_slab" class to create a list, "symm_ops", that contains classes,
            # which contain matrix and vector operators that may be used to rotate/translate the
            # slab about axes of symmetry
            symm_ops = sga_slab.get_symmetry_operations()
            # Create a boolean, "z_invertible", which will be "True" if the top of the slab is
            # the same as the bottom.
            z_invertible = True in map(lambda x: x.as_dict()['matrix'][2][2] == -1, symm_ops)
            # If the bottom is different, then...
            if not z_invertible:
                # flip the slab upside down...
                atoms_slab.rotate('x', math.pi, rotate_cell=True)

                # and if it is not in the database, then save it.
                slabdoc = mongo_doc(set_constraints(atoms_slab, len(atoms_slab)))
                tags = {'type':'slab',
                        'top':False,
                        'mpid':self.parameters['bulk']['mpid'],
                        'miller':self.parameters['slab']['miller'],
                        'shift':shift,
                        'num_slab_atoms':len(atoms_slab),
                        'relaxed':False,
                        'slab_generate_settings':self.parameters['slab']['slab_generate_settings'],
                        'get_slab_settings':self.parameters['slab']['get_slab_settings']}
                slabdoc['tags'] = tags
                slabsave.append(slabdoc)

        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(slabsave, open(self.temp_output_path, 'w'))

        return

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class GenerateAdsorbatesMarker(luigi.Task):
    parameters = luigi.DictParameter()

    def requires(self):
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return [GenerateSurfaces(parameters=OrderedDict(unrelaxed=True,
                                                            bulk=self.parameters['bulk'],
                                                            slab=self.parameters['slab'])),
                    GenerateBulk(parameters={'bulk':self.parameters['bulk']})]
        else:
            return [WriteRow(calctype='slab',
                             parameters=OrderedDict(bulk=self.parameters['bulk'],
                                                    slab=self.parameters['slab'])),
                    WriteRow(calctype='bulk',
                             parameters={'bulk':self.parameters['bulk']})]

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))

    def run(self):
        adsorbate = {'name':'U', 'atoms':Atoms('U')}

        slab_list = pickle.load(self.input()[0].open())
        bulk_atoms = mongo_doc_atoms(pickle.load(self.input()[1].open())[0])

        slabsave = []


        # For each slab, find all the adsorbates and add them to slabsave
        for slab in slab_list:

            # "bulk_atoms" [atoms class] and "slab_atoms" [atoms class] are the first bulk structure
            # and first slab structure, respectively, that we find in the database that corresponds
            # to the slab that we are looking at. Note that these lines ignore any possible repeats
            # of bulk or slab structures in the database.
            slab_atoms = mongo_doc_atoms(slab)
            # Initialize "sites" [list]
            sites = []

            # Repeat the atoms in the slab to get a cell that is at least as large as the "mix_xy"
            # parameter we set above.
            nx = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab_atoms.cell[0])))
            ny = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab_atoms.cell[1])))
            slabrepeat = (nx, ny, 1)
            slab_atoms.adsorbate_info = ''
            slab_atoms_repeat = slab_atoms.repeat(slabrepeat)

            # Work on each adsorbate-slab pair (for this particular slab). Note that "adsorbate"
            # will be a dictionary containing the atoms class and the name of the adsorbate.
            if len(slab_atoms_repeat) < self.parameters['adsorption']['max_slab_size']:
                sites = find_adsorption_sites(slab_atoms, bulk_atoms)
                site_list = sites

                # Work on each adsorption site for this adsorbate-slab pair
                for site in site_list:
                    if 'unrelaxed' in self.parameters:
                        shift = slab['tags']['shift']
                        top = slab['tags']['top']
                        miller = slab['tags']['miller']
                    else:
                        shift = self.parameters['slab']['shift']
                        top = self.parameters['slab']['top']
                        miller = self.parameters['slab']['miller']

                    # "tags" [dictionary] contains various information about the
                    # slab+adsorbate system
                    tags = {'type':'slab+adsorbate',
                            'adsorption_site':str(np.round(site, decimals=2)),
                            'slabrepeat':str(slabrepeat),
                            'adsorbate':adsorbate['name'],
                            'top':top,
                            'miller':miller,
                            'shift':shift,
                            'relaxed':False}

                    # Make a local copies of our "slab_atoms" and "adsorbate" classes...
                    slab_copy = slab_atoms_repeat.copy()
                    adsorbate_copy = adsorbate['atoms'].copy()
                    # Move the adsorbate onto the adsorption site...
                    adsorbate_copy.translate(site)
                    # Put the adsorbate onto the slab to create the "ads_atoms"
                    # adsorbate+slab system...
                    ads_atoms = slab_copy + adsorbate_copy
                    tags['atoms'] = ads_atoms

                    # and finally ad the system to the database along with some extra information.
                    slabsave.append(tags)

        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(slabsave, open(self.temp_output_path, 'w'))


class GenerateAdsorbates(luigi.Task):
    """
    This class takes a set of adsorbate positions from generate_adsorbates_marker and replaces
    the marker (a uranium atom) with the correct adsorbate. Adding an adsorbate is done in two
    steps (marker enumeration, then replacement) so that the hard work of enumerating all
    adsorption sites is only done once and reused for every adsorbate
    """
    parameters = luigi.DictParameter()

    def requires(self):
        """
        We need the generated adsorbates with the marker atoms.  We delete
        parameters['adsorption']['adsorbates'] so that every generate_adsorbates_marker call
        looks the same, even with different adsorbates requested in this task
        """
        parameters_no_adsorbate = copy.deepcopy(self.parameters)
        del parameters_no_adsorbate['adsorption']['adsorbates']
        return GenerateAdsorbatesMarker(parameters_no_adsorbate)

    def run(self):
        # Load the configurations
        adsorbate_configs = pickle.load(self.input().open())
        # For each configuration replace the marker with the adsorbate
        for adsorbate_config in adsorbate_configs:
            atoms = adsorbate_config['atoms']

            # Delete the marker
            adsorbate_position = atoms[-1].position
            del atoms[-1]

            # Load the adsorbate
            ads_atoms = pickle.loads(self.parameters['adsorption']['adsorbates'][0]['atoms'].decode('hex'))

            # Translate the adsorbate to the marker position and add it
            ads_atoms.translate(adsorbate_position)
            numslabatoms = len(atoms)
            atoms = atoms + ads_atoms

            # Set constraints and update the list of dictionaries with the correct atoms
            # object adsorbate name
            atoms.set_constraint()
            adsorbate_config['atoms'] = set_constraints(atoms, numslabatoms)
            adsorbate_config['adsorbate'] = self.parameters['adsorption']['adsorbates'][0]['name']

        # Save the generated list of adsorbate configurations to a pkl file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adsorbate_configs, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class CalculateEnergy(luigi.Task):
    """
    This class attempts to return the adsorption energy of a configuration relative to
    stoichiometric amounts of CO, H2, H2O
    """
    parameters = luigi.DictParameter()

    def requires(self):
        """
        We need the relaxed slab, the relaxed slab+adsorbate, and relaxed CO/H2/H2O gas
        structures/energies
        """
        toreturn = []
        toreturn.append(WriteRow(parameters=self.parameters, calctype='slab+adsorbate'))
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='', atoms=pickle.dumps(Atoms('')).encode('hex'))]
        toreturn.append(WriteRow(parameters=param, calctype='slab+adsorbate'))
        for gasname in ['CO', 'H2', 'H2O']:
            param = copy.deepcopy({'gas':self.parameters['gas']})
            param['gas']['gasname'] = gasname
            toreturn.append(WriteRow(parameters=param, calctype='gas'))
        # toreturn is a list of [slab+adsorbate,slab,CO,H2,H2O] relaxed structures
        return toreturn

    def run(self):
        inputs = self.input()

        # Load the gas phase energies
        gasEnergies = {}
        gasEnergies['CO'] = mongo_doc_atoms(pickle.load(inputs[2].open())[0]).get_potential_energy()
        gasEnergies['H2'] = mongo_doc_atoms(pickle.load(inputs[3].open())[0]).get_potential_energy()
        gasEnergies['H2O'] = mongo_doc_atoms(pickle.load(inputs[4].open())[0]).get_potential_energy()

        # Load the slab+adsorbate relaxed structures, and take the lowest energy one
        slab_ads = pickle.load(inputs[0].open())
        lowest_energy_slab = np.argmin(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_ads))
        slab_ads_energy = mongo_doc_atoms(slab_ads[lowest_energy_slab]).get_potential_energy()

        # Load the slab relaxed structures, and take the lowest energy one
        slab_blank = pickle.load(inputs[1].open())
        lowest_energy_blank = np.argmin(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_blank))
        slab_blank_energy = np.min(map(lambda x: mongo_doc_atoms(x).get_potential_energy(), slab_blank))

        # Get the per-atom energies as a linear combination of the basis set
        mono_atom_energies = {'H':gasEnergies['H2']/2.,
                              'O':gasEnergies['H2O']-gasEnergies['H2'],
                              'C':gasEnergies['CO']-(gasEnergies['H2O']-gasEnergies['H2'])}

        # Get the total energy of the stoichiometry amount of gas reference species
        gas_energy = 0
        for ads in self.parameters['adsorption']['adsorbates']:
            gas_energy += np.sum(map(lambda x: mono_atom_energies[x],
                                     adsorbate_dictionary(ads['name']).get_chemical_symbols())
                                )

        # Calculate the adsorption energy
        dE = slab_ads_energy - slab_blank_energy - gas_energy

        # Make an atoms object with a single-point calculator that contains the potential energy
        adjusted_atoms = mongo_doc_atoms(slab_ads[lowest_energy_slab])
        adjusted_atoms.set_calculator(SinglePointCalculator(adjusted_atoms,
                                                            forces=adjusted_atoms.get_forces(),
                                                            energy=dE))

        # Write a dictionary with the results and the entries that were used for the calculations
        # so that fwid/etc for each can be recorded
        towrite = {'atoms':adjusted_atoms,
                   'slab+ads':slab_ads[lowest_energy_slab],
                   'slab':slab_blank[lowest_energy_blank],
                   'gas':{'CO':pickle.load(inputs[2].open())[0],
                          'H2':pickle.load(inputs[3].open())[0],
                          'H2O':pickle.load(inputs[4].open())[0]}
                  }

        # Write the dictionary as a pickle
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(towrite, open(self.temp_output_path, 'w'))


    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


def fingerprint(atoms, siteind):
    """
    This function is used to fingerprint an atoms object
    atoms: atoms object to fingerprint
    siteind: the position of the binding atom in the adsorbate (assumed to be the first atom
    of the adsorbate)
    """

    # Delete the adsorbate except for the binding atom, then turn it into a uranium atom so we can
    # keep track of it in the coordination calculation
    atoms = atoms[0:siteind+1]
    atoms[-1].symbol = 'U'

    # Turn the atoms into a pymatgen structure file
    struct = AseAtomsAdaptor.get_structure(atoms[0:siteind+1])
    # PyMatGen [vcf class] of our system
    vcf = VoronoiCoordFinder(struct)
    # [list] of PyMatGen [periodic site class]es for each of the atoms that are
    # coordinated with the adsorbate
    coordinated_atoms = vcf.get_coordinated_sites(siteind, 0.8)
    # The elemental symbols for all of the coordinated atoms in a [list] of [unicode] objects
    coordinated_symbols = map(lambda x: x.species_string, coordinated_atoms)
    # Take out atoms that we assume are not part of the slab
    coordinated_symbols = [a for a in coordinated_symbols if a not in ['U']]
    # Turn the [list] of [unicode] values into a single [unicode]
    coordination = '-'.join(sorted(coordinated_symbols))
    # Make a [list] of human-readable coordination sites [unicode] for all of the slab atoms
    # that are coordinated to the adsorbate, "neighborcoord"
    neighborcoord = []
    for i in coordinated_atoms:
        # [int] that yields the slab+ads system's atomic index for the 1st-tier-coordinated atom
        neighborind = [site[0] for site in enumerate(struct) if i.distance(site[1]) < 0.1][0]
        # [list] of PyMatGen [periodic site class]es for each of the atoms that are coordinated
        # with the adsorbate
        coord = vcf.get_coordinated_sites(neighborind, 0.2)
        # The elemental symbols for all of the 2nd-tier-coordinated atoms in a [list] of
        # [unicode] objects
        coord_symbols = map(lambda x: x.species_string, coord)
        # Take out atoms that we assume are not part of the slab
        coord_symbols = [a for a in coord_symbols if a not in ['U']]
        # Sort the list of 2nd-tier-coordinated atoms to induce consistency
        coord_symbols.sort()
        # Turn the [list] of [unicode] values into a single [unicode]
        neighborcoord.append(i.species_string+':'+'-'.join(coord_symbols))
        
    # [list] of PyMatGen [periodic site class]es for each of the atoms that are
    # coordinated with the adsorbate
    coordinated_atoms_nextnearest = vcf.get_coordinated_sites(siteind, 0.2)
    # The elemental symbols for all of the coordinated atoms in a [list] of [unicode] objects
    coordinated_symbols_nextnearest = map(lambda x: x.species_string, coordinated_atoms_nextnearest)
    # Take out atoms that we assume are not part of the slab
    coordinated_symbols_nextnearest = [a for a in coordinated_symbols_nextnearest if a not in ['U']]
    # Turn the [list] of [unicode] values into a single [unicode]
    coordination_nextnearest = '-'.join(sorted(coordinated_symbols_nextnearest))

    # Return a dictionary with each of the fingerprints.  Any key/value pair can be added here
    # and will propagate up the chain
    return {'coordination':coordination, 'neighborcoord':neighborcoord, 'natoms':len(atoms),'nextnearestcoordination':coordination_nextnearest}


class FingerprintStructure(luigi.Task):
    """ This class fingerprints a relaxed structure """
    parameters = luigi.DictParameter()

    def requires(self):
        """
        We want the blank slab to know how many slab atoms there are, as well as the lowest-energy
        matching slab which calculateEnergy will get for us
        """
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='',
                                                         atoms=pickle.dumps(Atoms('')).
                                                         encode('hex'))]
        return [CalculateEnergy(self.parameters),
                WriteRow(parameters=param,
                         calctype='slab+adsorbate')]

    def run(self):

        # Load the lowest-energy configuration
        best_sys = pickle.load(self.input()[0].open())
        blank_entry = pickle.load(self.input()[1].open())

        # Make a human-readable adsorption site, "coordination" [unicode] (e.g., "Ag-Ag-Ag")
        initial_config = mongo_doc_atoms(best_sys['slab+ads']['initial_configuration'])

        expected_slab_atoms = blank_entry[0]['atoms']['natoms']

        if expected_slab_atoms == len(best_sys['atoms']):
            fp_final = {}
            fp_init={}
        else:
            # Calculate fingerprints for the initial and final state
            fp_final = fingerprint(best_sys['atoms'], expected_slab_atoms)
            fp_init = fingerprint(initial_config, expected_slab_atoms)

        # Save the the fingerprints of the final and initial state as a list in a pickle file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump([fp_final, fp_init], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class FingerprintGeneratedStructures(luigi.Task):
    """ This class is used to fingerprint non-relaxed structures """
    parameters = luigi.DictParameter()
    def requires(self):
        # Get the unrelaxed adsorbates and surfaces
        return [GenerateAdsorbates(self.parameters),
                GenerateSurfaces(parameters=OrderedDict(unrelaxed=True,
                                                         bulk=self.parameters['bulk'],
                                                         slab=self.parameters['slab']))]

    def run(self):
        # Load the lowest-energy configuration
        atomslist = pickle.load(self.input()[0].open())
        surfaces = pickle.load(self.input()[1].open())
        # Get the number of slab atoms
        if len(atomslist) > 0:
            expected_slab_atoms = len(surfaces[0]['atoms']['atoms'])*np.prod(eval(atomslist[0]['slabrepeat']))

            for entry in atomslist:
                if entry['adsorbate'] == '':
                    # We're looking at no adsorbates'
                    fp = {}
                else:
                    # Fingerprint the structure
                    fp = fingerprint(entry['atoms'], expected_slab_atoms)

                # Add the fingerprints to the dictionary
                for key in fp:
                    entry[key] = fp[key]

        # Write
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(atomslist, open(self.temp_output_path, 'w'))
    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class WriteAdsorptionConfig(luigi.Task):
    """
    This class is used to a processed adsorption calculation to a local ase-db database
    If you want the results to end up somewhere else, write a similar class and use it as the
    final target instead
    """
    parameters = luigi.DictParameter()

    def requires(self):
        """
        We want the lowest energy structure (with adsorption energy), the fingerprinted structure,
        and the bulk structure
        """
        return [CalculateEnergy(self.parameters),
                FingerprintStructure(self.parameters),
                WriteRow(calctype='bulk',
                         parameters={'bulk':self.parameters['bulk']})]

    def run(self):

        # Load the structure
        best_sys_pkl = pickle.load(self.input()[0].open())
        # Extract the atoms object
        best_sys = best_sys_pkl['atoms']
        # Get the lowest energy bulk structure
        bulk = pickle.load(self.input()[2].open())
        bulkmin = np.argmin(map(lambda x: x['results']['energy'], bulk))
        # Load the fingerprints of the initial and final state
        fingerprints = pickle.load(self.input()[1].open())
        fp_final = fingerprints[0]
        fp_init = fingerprints[1]
        for fp in [fp_init,fp_final]:
            for key in ['neighborcoord','coordination']:
                if key not in fp:
                    fp[key]=''

        # Make a dictionary of tags to add to the database
        criteria = {'type':'slab+adsorbate',
                    'mpid':self.parameters['bulk']['mpid'],
                    'miller':'(%d.%d.%d)'%(self.parameters['slab']['miller'][0],
                                           self.parameters['slab']['miller'][1],
                                           self.parameters['slab']['miller'][2]),
                    'num_slab_atoms':self.parameters['adsorption']['num_slab_atoms'],
                    'top':self.parameters['slab']['top'],
                    'slabrepeat':self.parameters['adsorption']['slabrepeat'],
                    'relaxed':True,
                    'adsorbate':self.parameters['adsorption']['adsorbates'][0]['name'],
                    'adsorption_site':self.parameters['adsorption']['adsorbates'][0]['adsorption_site'],
                    'coordination':fp_final['coordination'],
                    'nextnearestcoordination':fp_final['nextnearestcoordination'],
                    'neighborcoord':str(fp_final['neighborcoord']),
                    'initial_coordination':fp_init['coordination'],
                    'initial_nextnearestcoordination':fp_init['nextnearestcoordination'],
                    'initial_neighborcoord':str(fp_init['neighborcoord']),
                    'shift':self.parameters['slab']['shift'],
                    'fwid':best_sys_pkl['slab+ads']['fwid'],
                    'slabfwid':best_sys_pkl['slab']['fwid'],
                    'bulkfwid':bulk[bulkmin]['fwid']}
        # Turn the appropriate VASP tags into [str] so that ase-db may accept them.
        VSP_STNGS = vasp_settings_to_str(self.parameters['adsorption']['vasp_settings'])
        for key in VSP_STNGS:
            criteria[key] = VSP_STNGS[key]

        # Write the entry into the database
        with connect('adsorption_energy_database.db') as conAds:
            conAds.write(best_sys, **criteria)

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))


class WriteConfigsLocalDB(luigi.Task):
    """ This class is used to write a generated structure to a local ase database """
    parameters = luigi.DictParameter()

    def output(self):
        return luigi.LocalTarget('./structures/%s.pkl'%(self.task_id))

    def requires(self):
        """ Get the generated adsorbate configurations """
        return FingerprintGeneratedStructures(self.parameters)

    def run(self):
        with connect('enumerated_adsorption_sites.db') as con:

            # Load the configurations
            configs = pickle.load(self.input().open())
            # Find the unique configurations based on the fingerprint of each site
            unq_configs, unq_inds = np.unique(map(lambda x: str([x['shift'],
                                                                 x['coordination'],
                                                                 x['neighborcoord']]),
                                                  configs),
                                              return_index=True)
            # For each configuration, write a row to the dtabase
            for i in unq_inds:
                config = configs[i]
                con.write(config['atoms'],
                          shift=config['shift'],
                          miller=str(self.parameters['slab']['miller']),
                          mpid=self.parameters['bulk']['mpid'],
                          adsorbate=self.parameters['adsorption']['adsorbates'][0]['name'],
                          top=self.parameters['slab']['top'],
                          adsorption_site=config['adsorption_site'],
                          coordination=config['coordination'],
                          neighborcoord=str(config['neighborcoord']))   # Plus tags
        # Write a token file to indicate this task has been completed and added to the DB
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')


def default_parameter_slab(miller, top, shift, xc='beef-vdw', encut=350.):
    """ Generate some default parameters for a slab and expected relaxation settings """
    return OrderedDict(miller=miller,
                       top=top,
                       shift=shift,
                       relaxed=True,
                       vasp_settings=OrderedDict(xc=xc,
                                                 encut=encut,
                                                 ibrion=2,
                                                 nsw=100,
                                                 isif=0,
                                                 kpts=[4, 4, 1],
                                                 lreal='Auto',
                                                 ediffg=-0.03),
                       slab_generate_settings=OrderedDict(min_slab_size=7.,
                                                          min_vacuum_size=20.,
                                                          lll_reduce=False,
                                                          center_slab=True,
                                                          primitive=True,
                                                          max_normal_search=1),
                       get_slab_settings=OrderedDict(tol=0.3,
                                                     bonds=None,
                                                     max_broken_bonds=0,
                                                     symmetrize=False))


def default_parameter_gas(gasname, xc='beef-vdw', encut=350.):
    """ Generate some default parameters for a gas and expected relaxation settings """
    return OrderedDict(gasname=gasname,
                       relaxed=True,
                       vasp_settings=OrderedDict(xc=xc,
                                                 encut=encut,
                                                 ibrion=2,
                                                 nsw=100,
                                                 isif=0,
                                                 kpts=[1, 1, 1],
                                                 ediffg=-0.03))


def default_parameter_bulk(mpid, xc='beef-vdw', encutBulk=800.):
    """ Generate some default parameters for a bulk and expected relaxation settings """
    return OrderedDict(mpid=mpid,
                       relaxed=True,
                       vasp_settings=OrderedDict(xc=xc,
                                                 encut=encutBulk,
                                                 ibrion=1,
                                                 nsw=100,
                                                 isif=7,
                                                 ediff=1e-8,
                                                 kpts=[10, 10, 10],
                                                 prec='Accurate'))


def default_parameter_adsorption(adsorbate,
                                 adsorption_site=None,
                                 slabrepeat='(1, 1)',
                                 num_slab_atoms=0,
                                 xc='beef-vdw',
                                 encut=350.):
    """
    Generate some default parameters for an adsorption configuration and expected
    relaxation settings
    """

    adsorbateStructures = {'CO':{'atoms':Atoms('CO', positions=[[0.,0.,0.],[0.,0.,1.2]]),  'name':'CO'},
                           'H':{'atoms':Atoms('H',   positions=[[0.,0.,-0.5]]),            'name':'H'},
                           'O':{'atoms':Atoms('O',   positions=[[0.,0.,0.]]),              'name':'O'},
                           'C':{'atoms':Atoms('C',   positions=[[0.,0.,0.]]),              'name':'C'},
                           '':{'atoms':Atoms(),                                            'name':''},
                           'U':{'atoms':Atoms('U',   positions=[[0.,0.,0.]]),              'name':'U'},
                           'OH':{'atoms':Atoms('OH', positions=[[0.,0.,0.],[0.,0.,0.96]]), 'name':'OH'}}

    # This controls how many configurations get submitted if multiple configurations
    # match the criteria
    return OrderedDict(numtosubmit=1,
                       max_slab_size=60,
                       min_xy=4.5,
                       relaxed=True,
                       num_slab_atoms=num_slab_atoms,
                       slabrepeat=slabrepeat,
                       adsorbates=[OrderedDict(name=adsorbate,
                                               atoms=pickle.dumps(adsorbateStructures[adsorbate]['atoms']).encode('hex'),
                                               adsorption_site=adsorption_site)],
                       vasp_settings=OrderedDict(xc=xc,
                                                 encut=encut,
                                                 ibrion=2,
                                                 nsw=200,
                                                 isif=0,
                                                 kpts=[4, 4, 1],
                                                 lreal='Auto',
                                                 ediffg=-0.03))
