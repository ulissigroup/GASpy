'''
This module houses various functions and classes that Luigi uses to set up calculations that
can be submitted to Fireworks. This is intended to be used in conjunction with a submission
file, an example of which is named "adsorbtionTargets.py".
'''
# Python modules
import sys
from collections import OrderedDict
import cPickle as pickle
# 3rd party modules
import numpy as np
from ase.db import connect
import luigi
import dump
# GASpy modules
import fingerprint
sys.path.append('..')
from gaspy import defaults
from gaspy.utils import get_aux_db


LOCAL_DB_PATH = '/global/cscratch1/sd/zulissi/GASpy_DB/'


class AllDB(luigi.WrapperTask):
    '''
    First, dump from the Primary database to the Auxiliary database.
    Then, dump from the Auxiliary database to the Local adsorption energy database.
    Finally, re-request the adsorption energies to re-initialize relaxations & FW submissions.
    '''
    # write_db is a boolean. If false, we only execute FingerprintRelaxedAdslabs, which
    # submits calculations to Fireworks (if needed). If writeDB is true, then we still
    # exectute FingerprintRelaxedAdslabs, but we also dump to the Local DB.
    writeDB = luigi.BoolParameter(False)
    # max_processes is the maximum number of calculation sets to dump. If it's set to zero,
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
        dump.BulkGasToAuxDB().run()
        yield dump.SurfacesToAuxDB()

        # Get every row in the Aux database
        rows = get_aux_db().find({'type':'slab+adsorbate'})
        # Get all of the current fwid entries in the local DB
        with connect(LOCAL_DB_PATH+'/adsorption_energy_database.db') as enrg_db:
            fwids = [row.fwid for row in enrg_db.select()]

        # For each adsorbate/configuration, make a task to write the results to the output
        # database
        for i, row in enumerate(rows):
            # Break the loop if we reach the maxmimum number of processes
            if i+1 == self.max_processes:
                break

            # Only make the task if 1) the fireworks task is not already in the database,
            # 2) there is an adsorbate, and 3) we haven't reached the (non-zero) limit of rows
            # to dump.
            if (row['fwid'] not in fwids
                    and row['fwname']['adsorbate'] != ''
                    and ((self.max_processes == 0) or \
                         (self.max_processes > 0 and i < self.max_processes))):
                # Pull information from the Aux DB
                mpid = row['fwname']['mpid']
                miller = row['fwname']['miller']
                adsorption_site = row['fwname']['adsorption_site']
                adsorbate = row['fwname']['adsorbate']
                top = row['fwname']['top']
                num_slab_atoms = row['fwname']['num_slab_atoms']
                slabrepeat = row['fwname']['slabrepeat']
                shift = row['fwname']['shift']
                keys = ['gga', 'encut', 'zab_vdw', 'lbeefens', 'luse_vdw', 'pp', 'pp_version']
                settings = OrderedDict()
                for key in keys:
                    if key in row['fwname']['vasp_settings']:
                        settings[key] = row['fwname']['vasp_settings'][key]
                # Create the nested dictionary of information that we will store in the Aux DB
                parameters = {'bulk':defaults.bulk_parameters(mpid, settings=settings),
                              'gas':defaults.gas_parameters(gasname='CO', settings=settings),
                              'slab':defaults.slab_parameters(miller=miller,
                                                              shift=shift,
                                                              top=top,
                                                              settings=settings),
                              'adsorption':defaults.adsorption_parameters(adsorbate=adsorbate,
                                                                          num_slab_atoms=num_slab_atoms,
                                                                          slabrepeat=slabrepeat,
                                                                          adsorption_site=adsorption_site,
                                                                          settings=settings)}

                # Flag for hitting max_dump
                if i+1 == self.max_processes:
                    print('Reached the maximum number of processes, %s' % self.max_processes)
                # Dump to the local DB if we told Luigi to do so. We may do so by adding the
                # `--writeDB` flag when calling Luigi. If we do not dump to the local DB, then
                # we fingerprint the slab+adsorbate system
                if self.writeDB:
                    yield dump.ToLocalDB(parameters)
                else:
                    yield fingerprint.RelaxedAdslab(parameters)


class Enumerations(luigi.Task):
    '''
    This class re-requests the enumeration of adsorption sites to re-initialize our various
    generating functions. It then dumps any completed site enumerations into our Local DB
    for adsorption sites.
    '''
    parameters = luigi.DictParameter()

    def requires(self):
        ''' Get the generated adsorbate configurations '''
        return fingerprint.UnrelaxedAdslabs(self.parameters)

    def run(self):
        with connect(LOCAL_DB_PATH+'/enumerated_adsorption_sites.db') as con:
            # Load the configurations
            configs = pickle.load(self.input().open())
            # Find the unique configurations based on the fingerprint of each site
            unq_configs, unq_inds = np.unique(map(lambda x: str([x['shift'],
                                                                 x['coordination'],
                                                                 x['neighborcoord']]),
                                                  configs),
                                              return_index=True)
            # For each configuration, write a row to the database
            for i in unq_inds:
                config = configs[i]
                con.write(config['atoms'],
                          shift=config['shift'],
                          miller=str(self.parameters['slab']['miller']),
                          mpid=self.parameters['bulk']['mpid'],
                          adsorbate=self.parameters['adsorption']['adsorbates'][0]['name'],
                          top=config['top'],
                          adsorption_site=config['adsorption_site'],
                          coordination=config['coordination'],
                          neighborcoord=str(config['neighborcoord']),
                          nextnearestcoordination=str(config['nextnearestcoordination']))
        # Write a token file to indicate this task has been completed and added to the DB
        with self.output().temporary_path() as self.temp_output_path:
            with open(self.temp_output_path, 'w') as fhandle:
                fhandle.write(' ')

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
