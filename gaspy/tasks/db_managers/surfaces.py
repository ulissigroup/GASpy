'''
This module contains functions we use to update our `surface_energy` Mongo
collection, which contains surface energy calculations and associated
information.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']


#class DumpToSurfaceEnergyDB(luigi.Task):
#    ''' This class dumps the surface energies from our pickles to our tertiary databases '''
#    parameters = luigi.DictParameter()
#
#    def requires(self):
#        '''
#        We want the slabs at various thicknessed from CalculateSlabSurfaceEnergy and
#        the relevant bulk relaxation
#        '''
#        return [CalculateSlabSurfaceEnergy(self.parameters),
#                SubmitToFW(calctype='bulk',
#                           parameters={'bulk': self.parameters['bulk']})]
#
#    def run(self):
#
#        # Load the structures
#        surface_energy_pkl = pickle.load(open(self.input()[0].fn, 'rb'))
#
#        # Extract the atoms object
#        surface_energy_atoms = surface_energy_pkl[0]['atoms']
#
#        # Get the lowest energy bulk structure
#        bulk = pickle.load(open(self.input()[1].fn, 'rb'))
#        bulkmin = np.argmin([x['results']['energy'] for x in bulk])
#
#        # Calculate the movement for each relaxed slab
#        max_surface_movement = [utils.find_max_movement(doc['atoms'], make_atoms_from_doc(doc['initial_configuration']))
#                                for doc in surface_energy_pkl]
#
#        # Make a dictionary of tags to add to the database
#        processed_data = {'vasp_settings': self.parameters['slab']['vasp_settings'],
#                          'calculation_info': {'type': 'slab_surface_energy',
#                                               'formula': surface_energy_atoms.get_chemical_formula('hill'),
#                                               'mpid': self.parameters['bulk']['mpid'],
#                                               'miller': self.parameters['slab']['miller'],
#                                               'num_slab_atoms': len(surface_energy_atoms),
#                                               'relaxed': True,
#                                               'shift': surface_energy_pkl[0]['fwname']['shift']},
#                          'FW_info': surface_energy_pkl[0]['processed_data']['FW_info'],
#                          'surface_energy_info': surface_energy_pkl[0]['processed_data']['surface_energy_info'],
#                          'movement_data': {'max_surface_movement': max_surface_movement}}
#        processed_data['FW_info']['bulk'] = bulk[bulkmin]['fwid']
#        surface_energy_pkl_slab = make_doc_from_atoms(surface_energy_atoms)
#        surface_energy_pkl_slab['initial_configuration'] = surface_energy_pkl[0]['initial_configuration']
#        surface_energy_pkl_slab['processed_data'] = processed_data
#
#        # Write the entry into the database
#        with gasdb.get_mongo_collection('surface_energy') as collection:
#            collection.insert_one(surface_energy_pkl_slab)
#
#        # Write a blank token file to indicate this was done so that the entry is not written again
#        with self.output().temporary_path() as self.temp_output_path:
#            with open(self.temp_output_path, 'w') as fhandle:
#                fhandle.write(' ')
#
#    def output(self):
#        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
