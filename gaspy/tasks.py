'''
This module houses various tasks that Luigi uses to set up calculations that can be
submitted to Fireworks. This is intended to be used in conjunction with a bash submission
file.
'''

import pdb  # noqa: F401
import copy
import math
from datetime import datetime
from math import ceil
from collections import OrderedDict
import random
import cPickle as pickle
import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import rotate
from ase.collections import g2
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.matproj.rest import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from fireworks import Workflow
import luigi
from vasp.mongo import mongo_doc, mongo_doc_atoms
from gaspy import defaults, utils, gasdb
from gaspy import fireworks_helper_scripts as fwhs

# Get the path for the GASdb folder location from the gaspy config file
GASdb_path = utils.read_rc()['gasdb_path']


class GASpyTestTask(luigi.WrapperTask):
    def requires(self):
        print('Success!')


class UpdateAllDB(luigi.WrapperTask):
    '''
    First, dump from the Primary database to the Auxiliary database.
    Then, dump from the Auxiliary database to the Local adsorption energy database.
    Finally, re-request the adsorption energies to re-initialize relaxations & FW submissions.
    '''
    # max_processes is the maximum number of calculation sets to Dump If it's set to zero,
    # then there is no limit. This is used to limit the scope of a DB update for
    # debugging purposes.
    max_processes = luigi.IntParameter(0)

    def requires(self):
        '''
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        '''
        # Dump from the Primary DB to the Aux DB
        list(DumpToAuxDB().run())

        # Get every doc in the Aux database
        docs = gasdb.get_atoms_client().find({'type': 'slab+adsorbate'})
        # Get all of the current fwids numbers in the adsorption collection.
        # Turn the list into a dictionary so that we can parse through it faster.
        with gasdb.get_adsorption_client() as adsorption_client:
            fwids = [doc['processed_data']['FW_info']['slab+adsorbate'] for doc in adsorption_client.find()]
        fwids = dict.fromkeys(fwids)

        # For each adsorbate/configuration, make a task to write the results to the output
        # database. We also start a counter, `i`, for how many tasks we've processed.
        i = 0
        for doc in docs:
            # Only make the task if 1) the fireworks task is not already in the database, and
            # 2) there is an adsorbate
            if (doc['fwid'] not in fwids and doc['fwname']['adsorbate'] != ''):
                # Pull information from the Aux DB
                mpid = doc['fwname']['mpid']
                miller = doc['fwname']['miller']
                adsorption_site = doc['fwname']['adsorption_site']
                adsorbate = doc['fwname']['adsorbate']
                top = doc['fwname']['top']
                num_slab_atoms = doc['fwname']['num_slab_atoms']
                slabrepeat = doc['fwname']['slabrepeat']
                shift = doc['fwname']['shift']
                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
                settings = OrderedDict()
                for key in keys:
                    if key in doc['fwname']['vasp_settings']:
                        settings[key] = doc['fwname']['vasp_settings'][key]
                # Create the nested dictionary of information that we will store in the Aux DB
                parameters = {'bulk': defaults.bulk_parameters(mpid, settings=settings),
                              'gas': defaults.gas_parameters(gasname='CO', settings=settings),
                              'slab': defaults.slab_parameters(miller=miller,
                                                               shift=shift,
                                                               top=top,
                                                               settings=settings),
                              'adsorption': defaults.adsorption_parameters(adsorbate=adsorbate,
                                                                           num_slab_atoms=num_slab_atoms,
                                                                           slabrepeat=slabrepeat,
                                                                           adsorption_site=adsorption_site,
                                                                           settings=settings)}

                # If we've hit the maxmum number of processes, flag and stop
                i += 1
                if i == self.max_processes:
                    print('Reached the maximum number of processes, %s' % self.max_processes)
                    break

                yield DumpToAdsorptionDB(parameters)


class UpdateEnumerations(luigi.Task):
    '''
    This class re-requests the enumeration of adsorption sites to re-initialize our various
    generating functions. It then dumps any completed site enumerations into our Local DB
    for adsorption sites.
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        ''' Get the generated adsorbate configurations '''
        return FingerprintUnrelaxedAdslabs(self.parameters)

    def run(self):
        with gasdb.get_catalog_client() as catalog_client:
            # Load the configurations
            configs = pickle.load(self.input().open())
            # Find the unique configurations based on the fingerprint of each site
            unq_configs, unq_inds = np.unique(map(lambda x: str([x['shift'],
                                                                 x['fp']['coordination'],
                                                                 x['fp']['neighborcoord']]),
                                                  configs),
                                              return_index=True)
            # For each configuration, write a doc to the database
            for i in unq_inds:
                config = configs[i]
                atoms = config['atoms']
                slabadsdoc = mongo_doc(config['atoms'])
                processed_data = {'fp_init': config['fp'],
                                  'calculation_info': {'type': 'slab+adsorbate',
                                                       'formula': atoms.get_chemical_formula('hill'),
                                                       'mpid': self.parameters['bulk']['mpid'],
                                                       'miller': config['miller'],
                                                       'num_slab_atoms': len(atoms)-len(config['adsorbate']),
                                                       'top': config['top'],
                                                       'slabrepeat': config['slabrepeat'],
                                                       'relaxed': False,
                                                       'adsorbate': config['adsorbate'],
                                                       'shift': config['shift']}}
                slabadsdoc['processed_data'] = processed_data
                catalog_client.write(slabadsdoc)

        # Write a token file to indicate this task has been completed and added to the DB
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class DumpToAuxDB(luigi.Task):
    '''
    This class will load the results for the relaxations from the Primary FireWorks
    database into the Auxiliary vasp.mongo database.
    '''

    def run(self):
        lpad = fwhs.get_launchpad()

        # Get all of the FW numbers that have been loaded into the atoms collection already.
        # We turn the list into a dictionary so that we can parse through it more quickly.
        with gasdb.get_atoms_client() as atoms_client:
            atoms_fws = [a['fwid'] for a in atoms_client.find({'fwid': {'$exists': True}})]
            atoms_fws = dict.fromkeys(atoms_fws)

            # Get all of the completed fireworks from the Primary DB
            fws_cmpltd = lpad.get_fw_ids({'state': 'COMPLETED',
                                          'name.calculation_type': 'unit cell optimization'}) + \
                lpad.get_fw_ids({'state': 'COMPLETED',
                                 'name.calculation_type': 'gas phase optimization'}) + \
                lpad.get_fw_ids({'state': 'COMPLETED',
                                 'name.calculation_type': 'slab optimization',
                                 'name.shift': {'$exists': True}}) + \
                lpad.get_fw_ids({'state': 'COMPLETED',
                                 'name.calculation_type': 'slab+adsorbate optimization',
                                 'name.shift': {'$exists': True}})

            # For each fireworks object, turn the results into a mongo doc so that we can
            # dump the mongo doc into the Aux DB.
            for fwid in fws_cmpltd:
                if fwid not in atoms_fws:
                    # Get the information from the class we just pulled from the launchpad.
                    # Move on if we fail to get the info.
                    fw = lpad.get_fw_by_id(fwid)
                    try:
                        atoms, starting_atoms, trajectory, vasp_settings = fwhs.get_firework_info(fw)
                    except RuntimeError:
                        continue

                    # In an older version of GASpy, we did not use tags to identify
                    # whether an atom was part of the slab or an adsorbate. Here, we
                    # add the tags back in.
                    if (fw.created_on < datetime(2017, 7, 20) and
                            fw.name['calculation_type'] == 'slab+adsorbate optimization'):
                        # In this old version, the adsorbates were added onto the slab.
                        # Thus, the slab atoms came before the adsorbate atoms in
                        # the indexing. We use this information to create the tags list.
                        n_ads_atoms = len(fw.name['adsorbate'])
                        n_slab_atoms = len(atoms) - n_ads_atoms
                        tags = [0]*n_slab_atoms
                        tags.extend([1]*n_ads_atoms)
                        # Now set the tags for the atoms
                        atoms.set_tags(tags)
                        starting_atoms.set_tags(tags)

                    # Initialize the mongo document, doc, and the populate it with the fw info
                    doc = mongo_doc(atoms)
                    doc['initial_configuration'] = mongo_doc(starting_atoms)
                    doc['fwname'] = fw.name
                    doc['fwid'] = fwid
                    doc['directory'] = fw.launches[-1].launch_dir
                    if fw.name['calculation_type'] == 'unit cell optimization':
                        doc['type'] = 'bulk'
                    elif fw.name['calculation_type'] == 'gas phase optimization':
                        doc['type'] = 'gas'
                    elif fw.name['calculation_type'] == 'slab optimization':
                        doc['type'] = 'slab'
                    elif fw.name['calculation_type'] == 'slab+adsorbate optimization':
                        doc['type'] = 'slab+adsorbate'

                    # Convert the miller indices from strings to integers
                    if 'miller' in fw.name:
                        if isinstance(fw.name['miller'], str) or isinstance(fw.name['miller'], unicode):
                            doc['fwname']['miller'] = eval(doc['fwname']['miller'])
                    # Write the doc onto the Auxiliary database
                    atoms_client.write(doc)
                    print('Dumped a %s firework (FW ID %s) into the Auxiliary DB: ' % (doc['type'], fwid))
                    utils.print_dict(fw.name, indent=1)
                    # And while we're at it, dump the traj files, too
                    yield DumpFWToTraj(fwid=fwid)

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/DumpToAuxDB.token')


class DumpFWToTraj(luigi.Task):
    '''
    Given a FWID, this task will dump a traj file into GASdb/FW_structures for viewing/debugging
    purposes
    '''
    fwid = luigi.IntParameter()

    def run(self):
        lpad = fwhs.get_launchpad()
        fw = lpad.get_fw_by_id(self.fwid)
        atoms_hex = fw.launches[-1].action.stored_data['opt_results'][1]

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(atoms_hex.decode('hex'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/FW_structures/%s.traj' % (self.fwid))


class DumpToAdsorptionDB(luigi.Task):
    ''' This class dumps the adsorption energies from our pickles to our tertiary databases '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We want the lowest energy structure (with adsorption energy), the fingerprinted structure,
        and the bulk structure
        '''
        return [CalculateEnergy(self.parameters),
                FingerprintRelaxedAdslab(self.parameters),
                SubmitToFW(calctype='bulk',
                           parameters={'bulk': self.parameters['bulk']})]

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

        # Create and use tools to calculate the angle between the bond length of the diatomic
        # adsorbate and the z-direction of the bulk. We are not currently calculating triatomics
        # or larger.
        def unit_vector(vector):
            ''' Returns the unit vector of the vector.  '''
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            ''' Returns the angle in radians between vectors 'v1' and 'v2'::  '''
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if self.parameters['adsorption']['adsorbates'][0]['name'] in ['CO', 'OH']:
            angle = angle_between(best_sys[-1].position-best_sys[-2].position, best_sys.cell[2])
            if self.parameters['slab']['top'] is False:
                angle = np.abs(angle - math.pi)
        else:
            angle = 0.
        angle = angle/2./np.pi*360

        '''
        Calculate the maximum movement of surface atoms during the relaxation. then we do it again,
        but for adsorbate atoms.
        '''
        # First, calculate the number of adsorbate atoms
        num_adsorbate_atoms = len(pickle.loads(self.parameters['adsorption']['adsorbates'][0]['atoms'].decode('hex')))

        # An earlier version of GASpy added the adsorbate to the slab instead of the slab to the
        # adsorbate. Thus, the indexing for the slabs change. Here, we deal with that.
        lpad = fwhs.get_launchpad()
        fw = lpad.get_fw_by_id(best_sys_pkl['slab+ads']['fwid'])
        # *_start and *_end are the list indices to use when trying to pull out the * from
        # the adslab atoms.
        if fw.created_on < datetime(2017, 7, 20):
            slab_start = None
            slab_end = -num_adsorbate_atoms
            ads_start = -num_adsorbate_atoms
            ads_end = None
        else:
            slab_start = num_adsorbate_atoms
            slab_end = None
            ads_start = None
            ads_end = num_adsorbate_atoms
        # Get just the adslab's slab atoms in their initial and final state
        slab_initial = mongo_doc_atoms(best_sys_pkl['slab+ads']['initial_configuration'])[slab_start:slab_end]
        slab_final = best_sys[slab_start:slab_end]
        max_surface_movement = utils.find_max_movement(slab_initial, slab_final)
        # Repeat the procedure, but for adsorbates
        adsorbate_initial = mongo_doc_atoms(best_sys_pkl['slab+ads']['initial_configuration'])[ads_start:ads_end]
        adsorbate_final = best_sys[ads_start:ads_end]
        max_adsorbate_movement = utils.find_max_movement(adsorbate_initial, adsorbate_final)
        # Repeat the procedure, but for the relaxed bare slab
        bare_slab_initial = mongo_doc_atoms(best_sys_pkl['slab']['initial_configuration'])
        bare_slab_final = mongo_doc_atoms(best_sys_pkl['slab'])
        max_bare_slab_movement = utils.find_max_movement(bare_slab_initial, bare_slab_final)

        # Make a dictionary of tags to add to the database
        processed_data = {'fp_final': fp_final,
                          'fp_init': fp_init,
                          'vasp_settings': self.parameters['adsorption']['vasp_settings'],
                          'calculation_info': {'type': 'slab+adsorbate',
                                               'formula': best_sys.get_chemical_formula('hill'),
                                               'mpid': self.parameters['bulk']['mpid'],
                                               'miller': self.parameters['slab']['miller'],
                                               'num_slab_atoms': self.parameters['adsorption']['num_slab_atoms'],
                                               'top': self.parameters['slab']['top'],
                                               'slabrepeat': self.parameters['adsorption']['slabrepeat'],
                                               'relaxed': True,
                                               'adsorbates': self.parameters['adsorption']['adsorbates'],
                                               'adsorbate_names': map(lambda x: str(x['name']), self.parameters['adsorption']['adsorbates']),
                                               'shift': best_sys_pkl['slab+ads']['fwname']['shift']},
                          'FW_info': {'slab+adsorbate': best_sys_pkl['slab+ads']['fwid'],
                                      'slab': best_sys_pkl['slab']['fwid'],
                                      'bulk': bulk[bulkmin]['fwid']},
                          'movement_data': {'max_surface_movement': max_surface_movement,
                                            'max_adsorbate_movement': max_adsorbate_movement,
                                            'max_bare_slab_movement': max_bare_slab_movement}}
        best_sys_pkl_slab_ads = mongo_doc(best_sys_pkl['atoms'])
        best_sys_pkl_slab_ads['initial_configuration'] = best_sys_pkl['slab+ads']['initial_configuration']
        best_sys_pkl_slab_ads['processed_data'] = processed_data
        # Write the entry into the database

        with gasdb.get_adsorption_client() as adsorption_client:
            adsorption_client.write(best_sys_pkl_slab_ads)

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class SubmitToFW(luigi.Task):
    '''
    This class accepts a luigi.Task (e.g., relax a structure), then checks to see if
    this task is already logged in the Auxiliary vasp.mongo database. If it is not, then it
    submits the task to our Primary FireWorks database.
    '''
    # Calctype is one of 'gas','slab','bulk','slab+adsorbate'
    calctype = luigi.Parameter()

    # Parameters is a nested dictionary of parameters
    parameters = luigi.DictParameter()

    def requires(self):
        # Define a dictionary that will be used to search the Auxiliary database and find
        # the correct entry
        if self.calctype == 'gas':
            search_strings = {'type': 'gas',
                              'fwname.gasname': self.parameters['gas']['gasname']}
            for key in self.parameters['gas']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s' % key] = \
                    self.parameters['gas']['vasp_settings'][key]
        elif self.calctype == 'bulk':
            search_strings = {'type': 'bulk',
                              'fwname.mpid': self.parameters['bulk']['mpid']}
            for key in self.parameters['bulk']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s' % key] = \
                    self.parameters['bulk']['vasp_settings'][key]
        elif self.calctype == 'slab':
            search_strings = {'type': 'slab',
                              'fwname.miller': list(self.parameters['slab']['miller']),
                              'fwname.top': self.parameters['slab']['top'],
                              'fwname.shift': self.parameters['slab']['shift'],
                              'fwname.mpid': self.parameters['bulk']['mpid']}
            for key in self.parameters['slab']['vasp_settings']:
                if key not in ['isym']:
                    search_strings['fwname.vasp_settings.%s' % key] = \
                        self.parameters['slab']['vasp_settings'][key]
        elif self.calctype == 'slab+adsorbate':
            search_strings = {'type': 'slab+adsorbate',
                              'fwname.miller': list(self.parameters['slab']['miller']),
                              'fwname.top': self.parameters['slab']['top'],
                              'fwname.shift': self.parameters['slab']['shift'],
                              'fwname.mpid': self.parameters['bulk']['mpid'],
                              'fwname.adsorbate': self.parameters['adsorption']['adsorbates'][0]['name']}
            for key in self.parameters['adsorption']['vasp_settings']:
                if key not in ['nsw', 'isym', 'symprec']:
                    search_strings['fwname.vasp_settings.%s' % key] = \
                        self.parameters['adsorption']['vasp_settings'][key]
            if 'adsorption_site' in self.parameters['adsorption']['adsorbates'][0]:
                search_strings['fwname.adsorption_site'] = \
                    self.parameters['adsorption']['adsorbates'][0]['adsorption_site']
        # Round the shift to 4 decimal places so that we will be able to match shift numbers
        if 'fwname.shift' in search_strings:
            shift = search_strings['fwname.shift']
            search_strings['fwname.shift'] = {'$gte': shift - 1e-4, '$lte': shift + 1e-4}
            #search_strings['fwname.shift'] = np.cound(seach_strings['fwname.shift'], 4)

        # Grab all of the matching entries in the Auxiliary database
        with gasdb.get_atoms_client() as atoms_client:
            self.matching_doc = list(atoms_client.find(search_strings))
        #print('Search string:  %s', % search_strings)

        # If there are no matching entries, we need to yield a requirement that will
        # generate the necessary unrelaxed structure
        if len(self.matching_doc) == 0:
            if self.calctype == 'slab':
                return [GenerateSlabs(OrderedDict(bulk=self.parameters['bulk'],
                                                  slab=self.parameters['slab'])),
                        # We are also vaing the unrelaxed slabs just in case. We can delete if
                        # we can find shifts successfully.
                        GenerateSlabs(OrderedDict(unrelaxed=True,
                                                  bulk=self.parameters['bulk'],
                                                  slab=self.parameters['slab']))]
            if self.calctype == 'slab+adsorbate':
                # Return the base structure, and all possible matching ones for the surface
                search_strings = {'type': 'slab+adsorbate',
                                  'fwname.miller': list(self.parameters['slab']['miller']),
                                  'fwname.top': self.parameters['slab']['top'],
                                  'fwname.mpid': self.parameters['bulk']['mpid'],
                                  'fwname.adsorbate': self.parameters['adsorption']['adsorbates'][0]['name']}
                with gasdb.get_atoms_client() as atoms_client:
                    self.matching_docs_all_calcs = list(atoms_client.find(search_strings))

                #If we don't modify the parameters, the parameters will contain the FP for the request. This will trigger
                # a FingerprintUnrelaxedAdslabs for each FP, which is entirely unnecessary. The result is the same
                # regardless of what parameters['adsorption'][0]['fp'] happens to be

                parameters_copy = copy.deepcopy(self.parameters)
                if 'fp' in parameters_copy['adsorption']['adsorbates'][0]:
                    del parameters_copy['adsorption']['adsorbates'][0]['fp']

                return FingerprintUnrelaxedAdslabs(parameters_copy)

            if self.calctype == 'bulk':
                return GenerateBulk({'bulk': self.parameters['bulk']})
            if self.calctype == 'gas':
                return GenerateGas({'gas': self.parameters['gas']})

    def run(self):
        # If there are matching entries, this is easy, just dump the matching entries
        # into a pickle file
        if len(self.matching_doc) > 0:
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump(self.matching_doc, open(self.temp_output_path, 'w'))

        # Otherwise, we're missing a structure, so we need to submit whatever the
        # requirement returned
        else:
            launchpad = fwhs.get_launchpad()
            tosubmit = []

            # A way to append `tosubmit`, but specialized for gas relaxations
            if self.calctype == 'gas':
                name = {'vasp_settings': self.parameters['gas']['vasp_settings'],
                        'gasname': self.parameters['gas']['gasname'],
                        'calculation_type': 'gas phase optimization'}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       self.parameters['gas']['vasp_settings'],
                                                       max_atoms=self.parameters['bulk']['max_atoms']))

            # A way to append `tosubmit`, but specialized for bulk relaxations
            if self.calctype == 'bulk':
                name = {'vasp_settings': self.parameters['bulk']['vasp_settings'],
                        'mpid': self.parameters['bulk']['mpid'],
                        'calculation_type': 'unit cell optimization'}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       self.parameters['bulk']['vasp_settings'],
                                                       max_atoms=self.parameters['bulk']['max_atoms']))

            # A way to append `tosubmit`, but specialized for slab relaxations
            if self.calctype == 'slab':
                slab_docs = pickle.load(self.input()[0].open())
                atoms_list = [mongo_doc_atoms(slab_doc) for slab_doc in slab_docs
                              if float(np.round(slab_doc['tags']['shift'], 2)) ==
                              float(np.round(self.parameters['slab']['shift'], 2)) and
                              slab_doc['tags']['top'] == self.parameters['slab']['top']]
                if len(atoms_list) > 0:
                    #raise Exception('We found more than one slab that matches the shift')
                    #min_eng=np.argmin([atoms.get_potential_energy() for atoms in atoms_list])
                    #atoms=atoms_list[min_eng]
                    print('We found more than one slab that matches the shift. Just submitting the first one.')
                    atoms = atoms_list[0]
                elif len(atoms_list) == 0:
                    raise Exception('We did not find any slab that matches the shift: ' + str(self.parameters))
                elif len(atoms_list) == 1:
                    atoms = atoms_list[0]
                name = {'shift': self.parameters['slab']['shift'],
                        'mpid': self.parameters['bulk']['mpid'],
                        'miller': self.parameters['slab']['miller'],
                        'top': self.parameters['slab']['top'],
                        'vasp_settings': self.parameters['slab']['vasp_settings'],
                        'calculation_type': 'slab optimization',
                        'num_slab_atoms': len(atoms)}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       self.parameters['slab']['vasp_settings'],
                                                       max_atoms=self.parameters['bulk']['max_atoms'],
                                                       max_miller=self.parameters['slab']['max_miller']))

            # A way to append `tosubmit`, but specialized for adslab relaxations
            if self.calctype == 'slab+adsorbate':
                fpd_structs = pickle.load(self.input().open())

                def matchFP(entry, fp):
                    '''
                    This function checks to see if the first argument, `entry`, matches
                    a fingerprint, `fp`
                    '''
                    for key in fp:
                        if isinstance(entry[key], list):
                            if sorted(entry[key]) != sorted(fp[key]):
                                return False
                        else:
                            if entry[key] != fp[key]:
                                return False
                    return True
                # If there is an 'fp' key in parameters['adsorption']['adsorbates'][0], we
                # search for a site with the correct fingerprint, otherwise we search for an
                # adsorbate at the correct location
                if 'fp' in self.parameters['adsorption']['adsorbates'][0]:
                    matching_docs = [doc for doc in fpd_structs
                                     if matchFP(doc['fp'], self.parameters['adsorption']['adsorbates'][0]['fp'])]
                else:
                    if self.parameters['adsorption']['adsorbates'][0]['name'] != '':
                        matching_docs = [doc for doc in fpd_structs
                                         if doc['adsorption_site'] ==
                                         self.parameters['adsorption']['adsorbates'][0]['adsorption_site']]
                    else:
                        matching_docs = [doc for doc in fpd_structs]

                # If there is no adsorbate, then trim the matching_docs to the first doc we found.
                # Otherwise, trim the matching_docs to `numtosubmit`, a user-specified value that
                # decides the maximum number of fireworks that we want to submit.
                if self.parameters['adsorption']['adsorbates'][0]['name'] == '':
                    matching_docs = matching_docs[0:1]
                elif 'numtosubmit' in self.parameters['adsorption']:
                    matching_docs = matching_docs[0:self.parameters['adsorption']['numtosubmit']]

                # Add each of the matchig docs to `tosubmit`
                for doc in matching_docs:
                    # The name of our firework is actually a dictionary, as defined here
                    name = {'mpid': self.parameters['bulk']['mpid'],
                            'miller': self.parameters['slab']['miller'],
                            'top': self.parameters['slab']['top'],
                            'shift': doc['shift'],
                            'adsorbate': self.parameters['adsorption']['adsorbates'][0]['name'],
                            'adsorption_site': doc['adsorption_site'],
                            'vasp_settings': self.parameters['adsorption']['vasp_settings'],
                            'num_slab_atoms': self.parameters['adsorption']['num_slab_atoms'],
                            'slabrepeat': self.parameters['adsorption']['slabrepeat'],
                            'calculation_type': 'slab+adsorbate optimization'}
                    # If there is no adsorbate, then the 'adsorption_site' key is irrelevant
                    if name['adsorbate'] == '':
                        del name['adsorption_site']

                    '''
                    This next paragraph (i.e., code until the next blank line) is a prototyping
                    skeleton for GASpy Issue #14
                    '''
                    # First, let's see if we can find a reasonable guess for the doc:
                    # guess_docs=[doc2 for doc2 in self.matching_docs_all_calcs if matchFP(fingerprint(doc2['atoms'], ), doc)]
                    guess_docs = []
                    # We've found another calculation with exactly the same fingerprint
                    if len(guess_docs) > 0:
                        guess = guess_docs[0]
                        # Get the initial adsorption site of the identified doc
                        ads_site = np.array(map(eval, guess['fwname']['adsorption_site'].strip().split()[1:4]))
                        atoms = doc['atoms']
                        atomsguess = guess['atoms']
                        # For each adsorbate atom, move it the same relative amount as in the guessed configuration
                        lenAdsorbates = len(Atoms(self.parameters['adsorption']['adsorbates'][0]['name']))
                        for ind in range(-lenAdsorbates, len(atoms)):
                            atoms[ind].position += atomsguess[ind].position-ads_site
                    else:
                        atoms = doc['atoms']
                    if len(guess_docs) > 0:
                        name['guessed_from'] = {'xc': guess['fwname']['vasp_settings']['xc'],
                                                'encut': guess['fwname']['vasp_settings']['encut']}

                    # Add the firework if it's not already running
                    if len(fwhs.running_fireworks(name, launchpad)) == 0:
                        tosubmit.append(fwhs.make_firework(atoms, name,
                                                           self.parameters['adsorption']['vasp_settings'],
                                                           max_atoms=self.parameters['bulk']['max_atoms'],
                                                           max_miller=self.parameters['slab']['max_miller']))
                    # Filter out any blanks we may have introduced earlier, and then trim the
                    # number of submissions to our maximum.
                    tosubmit = [a for a in tosubmit if a is not None]
                    if 'numtosubmit' in self.parameters['adsorption']:
                        if len(tosubmit) > self.parameters['adsorption']['numtosubmit']:
                            tosubmit = tosubmit[0:self.parameters['adsorption']['numtosubmit']]
                            break

            # If we've found a structure that needs submitting, do so
            tosubmit = [a for a in tosubmit if a is not None]   # Trim blanks
            if len(tosubmit) > 0:
                wflow = Workflow(tosubmit, name='vasp optimization')
                launchpad.add_wf(wflow)
                print('Just submitted the following Fireworks: ')
                for fw in tosubmit:
                    utils.print_dict(fw.name, indent=1)

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class GenerateBulk(luigi.Task):
    '''
    This class pulls a bulk structure from Materials Project and then converts it to an ASE
    atoms object
    '''
    parameters = luigi.DictParameter()

    def run(self):
        # Connect to the Materials Project database
        with MPRester(utils.read_rc()['matproj_api_key']) as m:
            # Pull out the PyMatGen structure and convert it to an ASE atoms object
            structure = m.get_structure_by_material_id(self.parameters['bulk']['mpid'])
            atoms = AseAtomsAdaptor.get_atoms(structure)
            # Dump the atoms object into our pickles
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump([mongo_doc(atoms)], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class GenerateGas(luigi.Task):
    parameters = luigi.DictParameter()

    def run(self):
        atoms = g2[self.parameters['gas']['gasname']]
        atoms.positions += 10.
        atoms.cell = [20, 20, 20]
        atoms.pbc = [True, True, True]
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump([mongo_doc(atoms)], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class GenerateSlabs(luigi.Task):
    '''
    This class uses PyMatGen to create surfaces (i.e., slabs cut from a bulk) from ASE atoms
    objects
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        If the bulk does not need to be relaxed, we simply pull it from Materials Project using
        the `Bulk` class. If it needs to be relaxed, then we submit it to Fireworks.
        '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return GenerateBulk(parameters={'bulk': self.parameters['bulk']})
        else:
            return SubmitToFW(calctype='bulk', parameters={'bulk': self.parameters['bulk']})

    def run(self):
        # Preparation work with ASE and PyMatGen before we start creating the slabs
        bulk_doc = pickle.load(self.input().open())[0]
        # Pull out the fwid of the relaxed bulk (if there is one)
        if not ('unrelaxed' in self.parameters and self.parameters['unrelaxed']):
            bulk_fwid = bulk_doc['fwid']
        else:
            bulk_fwid = None
        bulk = mongo_doc_atoms(bulk_doc)
        structure = AseAtomsAdaptor.get_structure(bulk)
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        structure = sga.get_conventional_standard_structure()
        gen = SlabGenerator(structure,
                            self.parameters['slab']['miller'],
                            **self.parameters['slab']['slab_generate_settings'])
        slabs = gen.get_slabs(**self.parameters['slab']['get_slab_settings'])
        slabsave = []
        for slab in slabs:
            shift = slab.shift

            # Create an atoms class for this particular slab, "atoms_slab"
            atoms_slab = AseAtomsAdaptor.get_atoms(slab)
            # Then reorient the "atoms_slab" class so that the surface of the slab is pointing
            # upwards in the z-direction
            rotate(atoms_slab,
                   atoms_slab.cell[2], (0, 0, 1),
                   atoms_slab.cell[0], [1, 0, 0],
                   rotate_cell=True)
            # Save the slab, but only if it isn't already in the database
            top = True
            tags = {'type': 'slab',
                    'top': top,
                    'mpid': self.parameters['bulk']['mpid'],
                    'miller': self.parameters['slab']['miller'],
                    'shift': shift,
                    'num_slab_atoms': len(atoms_slab),
                    'relaxed': False,
                    'bulk_fwid': bulk_fwid,
                    'slab_generate_settings': self.parameters['slab']['slab_generate_settings'],
                    'get_slab_settings': self.parameters['slab']['get_slab_settings']}
            slabdoc = mongo_doc(utils.constrain_slab(atoms_slab))
            slabdoc['tags'] = tags
            slabsave.append(slabdoc)

            # If the top of the cut is not identical to the bottom, then save the bottom slab
            # to the database, as well. To do this, we first pull out the sga class of this
            # particular slab, "sga_slab". Again, we use a symmetry finding tolerance of 0.1
            # to be consistent with MP
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
                atoms_slab.wrap()
                atoms_slab.rotate('x', math.pi, rotate_cell=True, center='COM')
                if atoms_slab.cell[2][2] < 0.:
                    atoms_slab.cell[2] = -atoms_slab.cell[2]
                atoms_slab.wrap()

                # and if it is not in the database, then save it.
                slabdoc = mongo_doc(utils.constrain_slab(atoms_slab))
                tags = {'type': 'slab',
                        'top': not(top),
                        'mpid': self.parameters['bulk']['mpid'],
                        'miller': self.parameters['slab']['miller'],
                        'shift': shift,
                        'num_slab_atoms': len(atoms_slab),
                        'relaxed': False,
                        'slab_generate_settings': self.parameters['slab']['slab_generate_settings'],
                        'get_slab_settings': self.parameters['slab']['get_slab_settings']}
                slabdoc['tags'] = tags
                slabsave.append(slabdoc)

        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(slabsave, open(self.temp_output_path, 'w'))

        return

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class GenerateSiteMarkers(luigi.Task):
    '''
    This class will take a set of slabs, enumerate the adsorption sites on the slab, add a
    marker on the sites (i.e., Uranium), and then save the Uranium+slab systems into our
    pickles
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        If the system we are trying to create markers for is unrelaxed, then we only need
        to create the bulk and surfaces. If the system should be relaxed, then we need to
        submit the bulk and the slab to Fireworks.
        '''
        if 'unrelaxed' in self.parameters and self.parameters['unrelaxed']:
            return [GenerateSlabs(parameters=OrderedDict(unrelaxed=True,
                                                         bulk=self.parameters['bulk'],
                                                         slab=self.parameters['slab'])),
                    GenerateBulk(parameters={'bulk': self.parameters['bulk']})]
        else:
            return [SubmitToFW(calctype='slab',
                               parameters=OrderedDict(bulk=self.parameters['bulk'],
                                                      slab=self.parameters['slab'])),
                    SubmitToFW(calctype='bulk',
                               parameters={'bulk': self.parameters['bulk']})]

    def run(self):
        # Defire our marker, a uraniom Atoms object. Then pull out the slabs and bulk
        adsorbate = {'name': 'U', 'atoms': Atoms('U')}
        slab_docs = pickle.load(self.input()[0].open())
        bulk_doc = pickle.load(self.input()[1].open())[0]
        bulk = mongo_doc_atoms(bulk_doc)

        # Initialize `adslabs_to_save`, which will be a list containing marked slabs (i.e.,
        # adslabs) for us to save
        adslabs_to_save = []
        for slab_doc in slab_docs:
            # "slab" [atoms class] is the first slab structure in Aux DB that corresponds
            # to the slab that we are looking at. Note that thise any possible repeats of the
            # slab in the database.
            slab = mongo_doc_atoms(slab_doc)
            # Pull out the fwid of the relaxed slab (if there is one)
            if not ('unrelaxed' in self.parameters and self.parameters['unrelaxed']):
                slab_fwid = slab_doc['fwid']
            else:
                slab_fwid = None

            # Repeat the atoms in the slab to get a cell that is at least as large as the
            # "mix_xy" parameter we set above.
            nx = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[0])))
            ny = int(ceil(self.parameters['adsorption']['min_xy']/norm(slab.cell[1])))
            slabrepeat = (nx, ny, 1)
            slab.info['adsorbate_info'] = ''
            slab_repeat = slab.repeat(slabrepeat)

            # Find the adsorption sites. Then for each site we find, we create a dictionary
            # of tags to describe the site. Then we save the tags to our pickles.
            sites = utils.find_adsorption_sites(slab, bulk)
            for site in sites:
                # Populate the `tags` dictionary with various information
                if 'unrelaxed' in self.parameters:
                    shift = slab_doc['tags']['shift']
                    top = slab_doc['tags']['top']
                    miller = slab_doc['tags']['miller']
                else:
                    shift = self.parameters['slab']['shift']
                    top = self.parameters['slab']['top']
                    miller = self.parameters['slab']['miller']
                tags = {'type': 'slab+adsorbate',
                        'adsorption_site': str(np.round(site, decimals=2)),
                        'slabrepeat': str(slabrepeat),
                        'adsorbate': adsorbate['name'],
                        'top': top,
                        'miller': miller,
                        'shift': shift,
                        'slab_fwid': slab_fwid,
                        'relaxed': False}
                # Then add the adsorbate marker on top of the slab. Note that we use a local,
                # deep copy of the marker because the marker was created outside of this loop.
                _adsorbate = adsorbate['atoms'].copy()
                # Move the adsorbate onto the adsorption site...
                _adsorbate.translate(site)
                # Put the adsorbate onto the slab and add the adslab system to the tags
                adslab = slab_repeat.copy() + _adsorbate
                tags['atoms'] = adslab

                # Finally, add the information to list of things to save
                adslabs_to_save.append(tags)

        # Save the marked systems to our pickles
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adslabs_to_save, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class GenerateAdSlabs(luigi.Task):
    '''
    This class takes a set of adsorbate positions from SiteMarkers and replaces
    the marker (a uranium atom) with the correct adsorbate. Adding an adsorbate is done in two
    steps (marker enumeration, then replacement) so that the hard work of enumerating all
    adsorption sites is only done once and reused for every adsorbate
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We need the generated adsorbates with the marker atoms.  We delete
        parameters['adsorption']['adsorbates'] so that every generate_adsorbates_marker call
        looks the same, even with different adsorbates requested in this task
        '''
        parameters_no_adsorbate = copy.deepcopy(self.parameters)
        del parameters_no_adsorbate['adsorption']['adsorbates']
        return GenerateSiteMarkers(parameters_no_adsorbate)

    def run(self):
        # Load the configurations
        adsorbate_configs = pickle.load(self.input().open())

        # For each configuration replace the marker with the adsorbate
        for adsorbate_config in adsorbate_configs:
            # Load the atoms object for the slab and adsorbate
            slab = adsorbate_config['atoms']
            ads = pickle.loads(self.parameters['adsorption']['adsorbates'][0]['atoms'].decode('hex'))
            # Find the position of the marker/adsorbate and the number of slab atoms
            ads_pos = slab[-1].position
            # Delete the marker on the slab, and then put the slab under the adsorbate.
            # Note that we add the slab to the adsorbate in order to maintain any
            # constraints that may be associated with the adsorbate (because ase only
            # keeps the constraints of the first atoms object).
            del slab[-1]
            ads.translate(ads_pos)

            # If there is a hookean constraining the adsorbate to a local position, we need to adjust
            # it based on ads_pos. We only do this for hookean constraints fixed to a point
            for constraint in ads.constraints:
                dict_repr = constraint.todict()
                if dict_repr['name'] == 'Hookean' and constraint._type == 'point':
                    constraint.origin += ads_pos

            adslab = ads + slab
            adslab.cell = slab.cell
            adslab.pbc = [True, True, True]
            # We set the tags of slab atoms to 0, and set the tags of the adsorbate to 1.
            # In future version of GASpy, we intend to set the tags of co-adsorbates
            # to 2, 3, 4... etc (per co-adsorbate)
            tags = [1]*len(ads)
            tags.extend([0]*len(slab))
            adslab.set_tags(tags)
            # Set constraints for the slab and update the list of dictionaries with
            # the correct atoms object adsorbate name.
            adsorbate_config['atoms'] = utils.constrain_slab(adslab)
            adsorbate_config['adsorbate'] = self.parameters['adsorption']['adsorbates'][0]['name']

        # Save the generated list of adsorbate configurations to a pkl file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adsorbate_configs, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class MatchCatalogShift(luigi.Task):
    '''
    This class attempts to find the shift for slabs coming from a relaxed
    bulk structure that correspond to a slab and shift in the catalog
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        # We need both the relaxed and unrelaxed slabs to try and match them
        yield GenerateSlabs(parameters={'bulk': self.parameters['bulk'],
                                        'slab': self.parameters['slab'],
                                        'unrelaxed': True})
        yield GenerateSlabs(parameters={'bulk': self.parameters['bulk'],
                                        'slab': self.parameters['slab']})

    def run(self):
        # Pull out the mongo docs for both the unrelaxed and the relaxed slabs, respectively.
        all_cat_slab_docs = pickle.load(self.input()[0].open())
        cat_slab_docs = [slab for slab in all_cat_slab_docs
                         if np.abs(slab['tags']['shift']-self.parameters['slab']['shift'] < 0.01)]
        slab_docs = pickle.load(self.input()[1].open())

        # If it's 1-to-1, then assign the match and move on
        if len(slab_docs) == 1 and len(cat_slab_docs) == 1:
            shift = slab_docs[0]['tags']['shift']
        # If there are multiple potential matches, then use PyMatGen's `StructureMatcher`
        # class to figure out which relaxed structure corresponds to which catalog structure
        else:
            sm = StructureMatcher()
            cat_structure = AseAtomsAdaptor.get_structure(mongo_doc_atoms(cat_slab_docs[0]))
            structures = [AseAtomsAdaptor.get_structure(mongo_doc_atoms(doc)) for doc in slab_docs]
            scores = [sm.fit(cat_structure, structure) for structure in structures]
            match_index = np.argmin(scores)
            shift = slab_docs[match_index]['tags']['shift']

        # Save the matched shift
        with self.output().temporary_path() as self.temp_output_path:  # pylint:  disable=no-member
            pickle.dump(shift, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class FingerprintRelaxedAdslab(luigi.Task):
    '''
    This class takes relaxed structures from our Pickles, fingerprints them, then adds the
    fingerprints back to our Pickles
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        Our first requirement is CalculateEnergy, which relaxes the slab+ads system. Our second
        requirement is to relax the slab+ads system again, but without the adsorbates. We do
        this to ensure that the "blank slab" we are using in the adsorption calculations has
        the same number of slab atoms as the slab+ads system.
        '''
        # Here, we take the adsorbate off the slab+ads system
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='',
                                                         atoms=pickle.dumps(Atoms('')).encode('hex'))]
        return [CalculateEnergy(self.parameters),
                SubmitToFW(parameters=param,
                           calctype='slab+adsorbate')]

    def run(self):
        ''' We fingerprint the slab+adsorbate system both before and after relaxation. '''
        # Load the atoms objects for the lowest-energy slab+adsorbate (adslab) system and the
        # blank slab (slab)
        calc_e_dict = pickle.load(self.input()[0].open())
        slab = pickle.load(self.input()[1].open())

        # The atoms object for the adslab prior to relaxation
        adslab0 = mongo_doc_atoms(calc_e_dict['slab+ads']['initial_configuration'])
        # The number of atoms in the slab also happens to be the index for the first atom
        # of the adsorbate (in the adslab system)
        slab_natoms = slab[0]['atoms']['natoms']

        # If our "adslab" system actually doesn't have an adsorbate, then do not fingerprint
        if slab_natoms == len(calc_e_dict['atoms']):
            fp_final = {}
            fp_init = {}
        else:
            # Calculate fingerprints for the initial and final state
            fp_final = utils.fingerprint_atoms(calc_e_dict['atoms'])
            fp_init = utils.fingerprint_atoms(adslab0)

        # Save the the fingerprints of the final and initial state as a list in a pickle file
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump([fp_final, fp_init], open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class FingerprintUnrelaxedAdslabs(luigi.Task):
    '''
    This class takes unrelaxed slab+adsorbate (adslab) systems from our pickles, fingerprints
    the adslab, fingerprints the slab (without an adsorbate), and then adds fingerprints back
    to our Pickles. Note that we fingerprint the slab because we may have had to repeat the
    original slab to add the adsorbate onto it, and if so then we also need to fingerprint the
    repeated slab.
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We call the GenerateAdslabs class twice; once for the adslab, and once for the slab
        '''
        # Make a copy of `parameters` for our slab, but then we take off the adsorbate
        param_slab = copy.deepcopy(self.parameters)
        param_slab['adsorption']['adsorbates'] = \
            [OrderedDict(name='', atoms=pickle.dumps(Atoms('')).encode('hex'))]
        return [GenerateAdSlabs(self.parameters),
                GenerateAdSlabs(parameters=param_slab)]

    def run(self):
        # Load the list of slab+adsorbate (adslab) systems, and the bare slab. Also find the
        # number of slab atoms
        adslabs = pickle.load(self.input()[0].open())

        # Fingerprint each adslab
        for adslab in adslabs:
            # Don't bother if the adslab happens to be bare
            if adslab['adsorbate'] == '':
                fp = {}
            else:
                fp = utils.fingerprint_atoms(adslab['atoms'])
            # Add the fingerprints to the dictionary
            adslab['fp'] = fp

        # Write
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(adslabs, open(self.temp_output_path, 'w'))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class CalculateEnergy(luigi.Task):
    '''
    This class attempts to return the adsorption energy of a configuration relative to
    stoichiometric amounts of CO, H2, H2O
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We need the relaxed slab, the relaxed slab+adsorbate, and relaxed CO/H2/H2O gas
        structures/energies
        '''
        # Initialize the list of things that need to be done before we can calculate the
        # adsorption enegies
        toreturn = []

        # First, we need to relax the slab+adsorbate system
        toreturn.append(SubmitToFW(parameters=self.parameters, calctype='slab+adsorbate'))

        # Then, we need to relax the slab. We do this by taking the adsorbate off and
        # replacing it with '', i.e., nothing. It's still labeled as a 'slab+adsorbate'
        # calculation because of our code infrastructure.
        param = copy.deepcopy(self.parameters)
        param['adsorption']['adsorbates'] = [OrderedDict(name='', atoms=pickle.dumps(Atoms('')).encode('hex'))]
        toreturn.append(SubmitToFW(parameters=param, calctype='slab+adsorbate'))

        # Lastly, we need to relax the base gases.
        for gasname in ['CO', 'H2', 'H2O']:
            param = copy.deepcopy({'gas': self.parameters['gas']})
            param['gas']['gasname'] = gasname
            toreturn.append(SubmitToFW(parameters=param, calctype='gas'))

        # Now we put it all together.
        #print('Checking for/submitting relaxations for %s %s' % (self.parameters['bulk']['mpid'], self.parameters['slab']['miller']))
        return toreturn

    def run(self):
        inputs = self.input()

        # Load the gas phase energies
        gasEnergies = {}
        gasEnergies['CO'] = mongo_doc_atoms(pickle.load(inputs[2].open())[0]).get_potential_energy()
        gasEnergies['H2'] = mongo_doc_atoms(pickle.load(inputs[3].open())[0]).get_potential_energy()
        gasEnergies['H2O'] = mongo_doc_atoms(pickle.load(inputs[4].open())[0]).get_potential_energy()
        # Load the slab+adsorbate relaxed structures, and take the lowest energy one
        adslab_docs = pickle.load(inputs[0].open())
        lowest_energy_adslab = np.argmin(map(lambda doc: mongo_doc_atoms(doc).get_potential_energy(apply_constraint=False), adslab_docs))
        adslab_energy = mongo_doc_atoms(adslab_docs[lowest_energy_adslab]).get_potential_energy(apply_constraint=False)
        # Load the slab relaxed structures, and take the lowest energy one
        slab_docs = pickle.load(inputs[1].open())
        lowest_energy_slab = np.argmin(map(lambda doc: mongo_doc_atoms(doc).get_potential_energy(apply_constraint=False), slab_docs))
        slab_energy = np.min(map(lambda doc: mongo_doc_atoms(doc).get_potential_energy(apply_constraint=False), slab_docs))

        # Get the per-atom energies as a linear combination of the basis set
        mono_atom_energies = {'H': gasEnergies['H2']/2.,
                              'O': gasEnergies['H2O'] - gasEnergies['H2'],
                              'C': gasEnergies['CO'] - (gasEnergies['H2O']-gasEnergies['H2'])}

        # Get the total energy of the stoichiometry amount of gas reference species
        gas_energy = 0
        for ads in self.parameters['adsorption']['adsorbates']:
            gas_energy += np.sum(map(lambda x: mono_atom_energies[x],
                                     utils.ads_dict(ads['name']).get_chemical_symbols()))

        # Calculate the adsorption energy
        dE = adslab_energy - slab_energy - gas_energy

        # Make an atoms object with a single-point calculator that contains the potential energy
        adjusted_atoms = mongo_doc_atoms(adslab_docs[lowest_energy_adslab])
        adjusted_atoms.set_calculator(SinglePointCalculator(adjusted_atoms,
                                                            forces=adjusted_atoms.get_forces(apply_constraint=False),
                                                            energy=dE))

        # Write a dictionary with the results and the entries that were used for the calculations
        # so that fwid/etc for each can be recorded
        towrite = {'atoms': adjusted_atoms,
                   'slab+ads': adslab_docs[lowest_energy_adslab],
                   'slab': slab_docs[lowest_energy_slab],
                   'gas': {'CO': pickle.load(inputs[2].open())[0],
                           'H2': pickle.load(inputs[3].open())[0],
                           'H2O': pickle.load(inputs[4].open())[0]}}

        # Write the dictionary as a pickle
        with self.output().temporary_path() as self.temp_output_path:
            pickle.dump(towrite, open(self.temp_output_path, 'w'))

        for ads in self.parameters['adsorption']['adsorbates']:
            print('Finished CalculateEnergy for %s on the %s site of %s %s:  %s eV' % (ads['name'],
                  self.parameters['adsorption']['adsorbates'][0]['adsorption_site'],
                  self.parameters['bulk']['mpid'],
                  self.parameters['slab']['miller'],
                  dE))

    def output(self):
        return luigi.LocalTarget(GASdb_path+'/pickles/%s.pkl' % (self.task_id))


class EnumerateAlloys(luigi.WrapperTask):
    '''
    This class is meant to be called by Luigi to begin relaxations of a database of alloys
    '''
    max_index = luigi.IntParameter(1)
    whitelist = luigi.ListParameter()
    max_to_submit = luigi.IntParameter(1000)

    def requires(self):
        """
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        """
        # Define some elements that we don't want alloys with (note no oxides for the moment)
        all_elements = ['H', 'He', 'Li', 'Be', 'B', 'C',
                        'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
                        'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
                        'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                        'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                        'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                        'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
                        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                        'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                        'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uuq', 'Uuh']
        whitelist = self.whitelist
        restricted_elements = [el for el in all_elements if el not in whitelist]

        # Query MP for all alloys that are stable, near the lower hull, and don't have one of the
        # restricted elements
        with MPRester("MGOdX3P4nI18eKvE") as m:
            results = m.query({"elements": {"$nin": restricted_elements},
                               "e_above_hull": {"$lt": 0.1},
                               "formation_energy_per_atom": {"$lte": 0.0}},
                              ['pretty_formula',
                               'formula',
                               'spacegroup',
                               'material id',
                               'taskid',
                               'task_id',
                               'structure'],
                              mp_decode=True)

        # Define how to enumerate all of the facets for a given material
        def processStruc(result):
            struct = result['structure']
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            structure = sga.get_conventional_standard_structure()
            miller_list = get_symmetrically_distinct_miller_indices(structure, self.max_index)
            # pickle.dump(structure,open('./bulks/%s.pkl'%result['task_id'],'w'))
            return map(lambda x: [result['task_id'], x], miller_list)

        # Generate all facets for each material in parallel
        all_miller = utils.multimap(processStruc, results)

        print('Total # of matching surfaces in MP: %d' % (np.sum([len(x) for x in all_miller])))

        tasks_to_submit = []
        for facets in reversed(all_miller):
            for facet in facets:
                task = UpdateEnumerations(parameters=OrderedDict(unrelaxed=True,
                                                                 bulk=defaults.bulk_parameters(facet[0], max_atoms = 50),
                                                                 slab=defaults.slab_parameters(facet[1], True, 0),
                                                                 gas=defaults.gas_parameters('CO'),
                                                                 adsorption=defaults.adsorption_parameters('U',
                                                                                                           '[3.36 1.16 24.52]',
                                                                                                           '(1, 1)', 24)))
                if not(task.complete()):
                    tasks_to_submit.append(task)

        random.shuffle(tasks_to_submit)

        if len(tasks_to_submit) > self.max_to_submit:
            tasks_to_submit = tasks_to_submit[0: self.max_to_submit]

        return tasks_to_submit

        #    for i in range(self.max_to_submit):
        #        yield tasks_to_submit
        #            submitted+=1
        #            yield t
        #            if submitted > self.max_to_submit:
        #                return
