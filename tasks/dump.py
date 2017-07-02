# Python modules
import sys
from collections import OrderedDict
import cPickle as pickle
import math
# 3rd part modules
import numpy as np
from ase.geometry import find_mic
from ase.db import connect
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_analyzer import average_coordination_number
from vasp.mongo import mongo_doc, mongo_doc_atoms
import luigi
# GASpy modules
import generate
import calculate
import fingerprint
from submit_to_fw import SubmitToFW
sys.path.append('..')
from gaspy import defaults
from gaspy.utils import get_aux_db, print_dict, vasp_settings_to_str
from gaspy.fireworks_helper_scripts import get_launchpad, get_firework_info


LOCAL_DB_PATH = '/global/cscratch1/sd/zulissi/GASpy_DB/'


class BulkGasToAuxDB(luigi.Task):
    '''
    This class will load the results for bulk and slab relaxations from the Primary FireWorks
    database into the Auxiliary vasp.mongo database.
    '''

    def run(self):
        lpad = get_launchpad()

        # Create a class, "con", that has methods to interact with the database.
        with get_aux_db() as aux_db:

            # A list of integers containing the Fireworks job ID numbers that have been
            # added to the database already
            fws = [a['fwid'] for a in aux_db.find({'fwid':{'$exists':True}})]

            # Get all of the completed fireworks for unit cells and gases
            fws_cmpltd = lpad.get_fw_ids({'state':'COMPLETED',
                                          'name.calculation_type':'unit cell optimization'}) + \
                         lpad.get_fw_ids({'state':'COMPLETED',
                                          'name.calculation_type':'gas phase optimization'})

            # For each fireworks object, turn the results into a mongo doc
            for fwid in fws_cmpltd:
                if fwid not in fws:
                    # Get the information from the class we just pulled from the launchpad
                    fw = lpad.get_fw_by_id(fwid)
                    atoms, starting_atoms, trajectory, vasp_settings = get_firework_info(fw)

                    # Initialize the mongo document, doc, and the populate it with the fw info
                    doc = mongo_doc(atoms)
                    doc['initial_configuration'] = mongo_doc(starting_atoms)
                    doc['fwname'] = fw.name
                    doc['fwid'] = fwid
                    doc['directory'] = fw.launches[-1].launch_dir
                    # fw.name['vasp_settings'] = vasp_settings
                    if fw.name['calculation_type'] == 'unit cell optimization':
                        doc['type'] = 'bulk'
                    elif fw.name['calculation_type'] == 'gas phase optimization':
                        doc['type'] = 'gas'
                    # Convert the miller indices from strings to integers
                    if 'miller' in fw.name:
                        if isinstance(fw.name['miller'], str) \
                        or isinstance(fw.name['miller'], unicode):
                            doc['fwname']['miller'] = eval(doc['fwname']['miller'])

                    # Write the doc onto the Auxiliary database
                    aux_db.write(doc)
                    print('Dumped a %s firework (FW ID %s) into the Auxiliary DB:' \
                          % (doc['type'], fwid))
                    print_dict(fw.name, indent=1)


class SurfacesToAuxDB(luigi.Task):
    '''
    This class will load the results for surface relaxations from the Primary FireWorks database
    into the Auxiliary vasp.mongo database.
    '''

    def requires(self):
        lpad = get_launchpad()

        # A list of integers containing the Fireworks job ID numbers that have been
        # added to the database already
        with get_aux_db() as aux_db:
            fws = [a['fwid'] for a in aux_db.find({'fwid':{'$exists':True}})]

        # Get all of the completed fireworks for slabs and slab+ads
        fws_cmpltd = lpad.get_fw_ids({'state':'COMPLETED',
                                      'name.calculation_type':'slab optimization'}) + \
                     lpad.get_fw_ids({'state':'COMPLETED',
                                      'name.calculation_type':'slab+adsorbate optimization'})

        # Trouble-shooting code
        #random.seed(42)
        #random.shuffle(fws_cmpltd)
        #fws_cmpltd=fws_cmpltd[-60:]
        fws_cmpltd.reverse()

        # `surfaces` will be a list of the different surfaces that we need to generate before
        # we are able to dump them to the Auxiliary DB.
        surfaces = []
        # `to_dump` will be a list of lists. Each sublist contains information we need to dump
        # a surface from the Primary DB to the Auxiliary DB
        self.to_dump = []
        self.missing_shift_to_dump = []

        # For each fireworks object, turn the results into a mongo doc
        for fwid in fws_cmpltd:
            if fwid not in fws:
                # Get the information from the class we just pulled from the launchpad
                fw = lpad.get_fw_by_id(fwid)
                atoms, starting_atoms, trajectory, vasp_settings = get_firework_info(fw)
                # Prepare to add VASP settings to the doc
                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
                settings = OrderedDict()
                for key in keys:
                    if key in vasp_settings:
                        settings[key] = vasp_settings[key]
                # Convert the miller indices from strings to integers
                if isinstance(fw.name['miller'], str) or isinstance(fw.name['miller'], unicode):
                    miller = eval(fw.name['miller'])
                else:
                    miller = fw.name['miller']
                #print(fw.name['mpid'])

                '''
                This next paragraph of code (i.e., the lines until the next blank line)
                addresses our old results that were saved without shift values. Here, we
                re-create a surface so that we can guess what its shift is later on.
                '''
                # Create the surfaces
                if 'shift' not in fw.name:
                    surfaces.append(generate.Slabs({'bulk': defaults.bulk_parameters(mpid=fw.name['mpid'],
                                                                                     settings=settings),
                                                    'slab': defaults.slab_parameters(miller=miller,
                                                                                     top=True,
                                                                                     shift=0.,
                                                                                     settings=settings)}))
                    self.missing_shift_to_dump.append([atoms, starting_atoms, trajectory,
                                                       vasp_settings, fw, fwid])
                else:

                    # Pass the list of surfaces to dump to `self` so that it can be called by the
                    #`run' method
                    self.to_dump.append([atoms, starting_atoms, trajectory,
                                         vasp_settings, fw, fwid])

        # Establish that we need to create the surfaces before dumping them
        return surfaces

    def run(self):
        selfinput = self.input()

        # Create a class, "aux_db", that has methods to interact with the database.
        with get_aux_db() as aux_db:

            # Start a counter for how many surfaces we will be guessing shifts for
            n_missing_shift = 0

            # Pull out the information for each surface that we put into to_dump
            for atoms, starting_atoms, trajectory, vasp_settings, fw, fwid \
                in self.missing_shift_to_dump + self.to_dump:
                # Initialize the mongo document, doc, and the populate it with the fw info
                doc = mongo_doc(atoms)
                doc['initial_configuration'] = mongo_doc(starting_atoms)
                doc['fwname'] = fw.name
                doc['fwid'] = fwid
                doc['directory'] = fw.launches[-1].launch_dir
                if fw.name['calculation_type'] == 'slab optimization':
                    doc['type'] = 'slab'
                elif fw.name['calculation_type'] == 'slab+adsorbate optimization':
                    doc['type'] = 'slab+adsorbate'
                # Convert the miller indices from strings to integers
                if 'miller' in fw.name:
                    if isinstance(fw.name['miller'], str) or isinstance(fw.name['miller'], unicode):
                        doc['fwname']['miller'] = eval(doc['fwname']['miller'])

                '''
                This next paragraph of code (i.e., the lines until the next blank line)
                addresses our old results that were saved without shift values. Here, we
                guess what the shift is (based on information from the surface we created before
                in the "requires" function) and declare it before saving it to the database.
                '''
                if 'shift' not in doc['fwname']:
                    slab_list_unrelaxed = pickle.load(selfinput[n_missing_shift].open())
                    n_missing_shift += 1
                    atomlist_unrelaxed = [mongo_doc_atoms(slab)
                                          for slab in slab_list_unrelaxed
                                          if slab['tags']['top'] == fw.name['top']]
                    if len(atomlist_unrelaxed) > 1:
                        #pprint(atomlist_unrelaxed)
                        #pprint(fw)
                        # We use the average coordination as a descriptor of the structure,
                        # there should be a pretty large change with different shifts
                        def getCoord(x):
                            return average_coordination_number([AseAtomsAdaptor.get_structure(x)])
                        # Get the coordination for the unrelaxed surface w/ correct shift
                        if doc['type'] == 'slab':
                            reference_coord = getCoord(starting_atoms)
                        elif doc['type'] == 'slab+adsorbate':
                            try:
                                num_adsorbate_atoms = {'':0, 'OH':2, 'CO':2, 'C':1, 'H':1, 'O':1}[fw.name['adsorbate']]
                            except KeyError:
                                print("%s is not recognizable by GASpy's adsorbates dictionary. \
                                      Please add it to `num_adsorbate_atoms` \
                                      in `dump.SurfacesToAuxDB`" % fw.name['adsorbate'])
                            if num_adsorbate_atoms > 0:
                                starting_blank = starting_atoms[0:-num_adsorbate_atoms]
                            else:
                                starting_blank = starting_atoms
                            reference_coord = getCoord(starting_blank)
                        # Get the coordination for each unrelaxed surface
                        unrelaxed_coord = map(getCoord, atomlist_unrelaxed)
                        # We want to minimize the distance in these dictionaries
                        def getDist(x, y):
                            vals = []
                            for key in x:
                                vals.append(x[key]-y[key])
                            return np.linalg.norm(vals)
                        # Get the distances to the reference coordinations
                        dist = map(lambda x: getDist(x, reference_coord), unrelaxed_coord)
                        # Grab the atoms object that minimized this distance
                        shift = slab_list_unrelaxed[np.argmin(dist)]['tags']['shift']
                        doc['fwname']['shift'] = float(np.round(shift, 4))
                        doc['fwname']['shift_guessed'] = True
                    else:
                        doc['fwname']['shift'] = 0
                        doc['fwname']['shift_guessed'] = True

                aux_db.write(doc)
                print('Dumped a %s firework (FW ID %s) into the Auxiliary DB:' \
                      % (doc['type'], fwid))
                print_dict(fw.name, indent=1)

        # Touch the token to indicate that we've written to the database
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/DumpToAuxDB.token')


class ToLocalDB(luigi.Task):
    ''' This class dumps the adsorption energies from our pickles to our Local energies DB '''
    parameters = luigi.DictParameter()

    def requires(self):
        '''
        We want the lowest energy structure (with adsorption energy), the fingerprinted structure,
        and the bulk structure
        '''
        return [calculate.Energy(self.parameters),
                fingerprint.RelaxedAdslab(self.parameters),
                SubmitToFW(calctype='bulk',
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
        for fp in [fp_init, fp_final]:
            for key in ['neighborcoord', 'nextnearestcoordination', 'coordination']:
                if key not in fp:
                    fp[key] = ''

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
        # Get just the slab atoms of the initial and final state
        slab_atoms_final = best_sys[0:-num_adsorbate_atoms]
        slab_atoms_initial = mongo_doc_atoms(best_sys_pkl['slab+ads']['initial_configuration'])[0:-num_adsorbate_atoms]
        # Calculate the distances for each atom
        distances = slab_atoms_final.positions - slab_atoms_initial.positions
        # Reduce the distances in case atoms wrapped around (the minimum image convention)
        dist, Dlen = find_mic(distances, slab_atoms_final.cell, slab_atoms_final.pbc)
        # Calculate the max movement of the surface atoms
        max_surface_movement = np.max(Dlen)
        # Repeat the procedure, but for adsorbates
        # get just the slab atoms of the initial and final state
        adsorbate_atoms_final = best_sys[-num_adsorbate_atoms:]
        adsorbate_atoms_initial = mongo_doc_atoms(best_sys_pkl['slab+ads']['initial_configuration'])[-num_adsorbate_atoms:]
        distances = adsorbate_atoms_final.positions - adsorbate_atoms_initial.positions
        dist, Dlen = find_mic(distances, slab_atoms_final.cell, slab_atoms_final.pbc)
        max_adsorbate_movement = np.max(Dlen)

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
                    'shift':best_sys_pkl['slab+ads']['fwname']['shift'],
                    'fwid':best_sys_pkl['slab+ads']['fwid'],
                    'slabfwid':best_sys_pkl['slab']['fwid'],
                    'bulkfwid':bulk[bulkmin]['fwid'],
                    'adsorbate_angle':angle,
                    'max_surface_movement':max_surface_movement,
                    'max_adsorbate_movement':max_adsorbate_movement}
        # Turn the appropriate VASP tags into [str] so that ase-db may accept them.
        VSP_STNGS = vasp_settings_to_str(self.parameters['adsorption']['vasp_settings'])
        for key in VSP_STNGS:
            if key == 'pp_version':
                criteria[key] = VSP_STNGS[key] + '.'
            else:
                criteria[key] = VSP_STNGS[key]

        # Write the entry into the database
        with connect(LOCAL_DB_PATH+'/adsorption_energy_database.db') as conAds:
            conAds.write(best_sys, **criteria)

        # Write a blank token file to indicate this was done so that the entry is not written again
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
