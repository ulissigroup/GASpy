'''
This module houses the core functions needed to submit rockets to FireWorks
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import copy
from collections import OrderedDict
import pickle
import numpy as np
import luigi
from fireworks import Workflow
from ...mongo import make_atoms_from_doc
from ... import utils, gasdb, defaults
from ... import fireworks_helper_scripts as fwhs

GASDB_PATH = utils.read_rc('gasdb_path')


class SubmitToFW(luigi.Task):
    '''
    This class accepts a luigi.Task (e.g., relax a structure), then checks to see if
    this task is already logged in the Auxiliary vasp.mongo database. If it is not, then it
    submits the task to our Primary FireWorks database.
    '''
    # Calctype is one of 'gas', 'slab', 'bulk', 'slab+adsorbate'
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

        elif self.calctype == 'slab_surface_energy':
            # pretty much identical to "slab" above, except no top since top/bottom
            # surfaces are both relaxed
            search_strings = {'type': 'slab_surface_energy',
                              'fwname.miller': list(self.parameters['slab']['miller']),
                              'fwname.num_slab_atoms': self.parameters['slab']['natoms'],
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
            if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
                for key in self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation']:
                    search_strings['fwname.adsorbate_rotation.%s' % key] = \
                        self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation'][key]

        # Round the shift to 4 decimal places so that we will be able to match shift numbers
        if 'fwname.shift' in search_strings:
            shift = search_strings['fwname.shift']
            search_strings['fwname.shift'] = {'$gte': shift - 1e-4, '$lte': shift + 1e-4}

        # Grab all of the matching entries in the Auxiliary database
        with gasdb.get_mongo_collection('atoms') as collection:
            self.matching_doc = list(collection.find(search_strings))

        # If there are no matching entries, we need to yield a requirement that will
        # generate the necessary unrelaxed structure
        if len(self.matching_doc) == 0:
            if self.calctype == 'slab':
                from ..structure_generators import GenerateSlabs
                return [GenerateSlabs(OrderedDict(bulk=self.parameters['bulk'],
                                                  slab=self.parameters['slab'])),
                        # We are also vaing the unrelaxed slabs just in case. We can delete if
                        # we can find shifts successfully.
                        GenerateSlabs(OrderedDict(unrelaxed=True,
                                                  bulk=self.parameters['bulk'],
                                                  slab=self.parameters['slab']))]
            if self.calctype == 'slab_surface_energy':
                from ..structure_generators import GenerateSlabs
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
                with gasdb.get_mongo_collection('atoms') as collection:
                    self.matching_docs_all_calcs = list(collection.find(search_strings))

                # If we don't modify the parameters, the parameters will contain the FP for the
                # request. This will trigger a FingerprintUnrelaxedAdslabs for each FP, which
                # is entirely unnecessary. The result is the same # regardless of what
                # parameters['adsorption'][0]['fp'] happens to be
                from ..metadata_calculators import FingerprintUnrelaxedAdslabs
                parameters_copy = utils.unfreeze_dict(copy.deepcopy(self.parameters))
                if 'fp' in parameters_copy['adsorption']['adsorbates'][0]:
                    del parameters_copy['adsorption']['adsorbates'][0]['fp']
                parameters_copy['unrelaxed'] = 'relaxed_bulk'
                return FingerprintUnrelaxedAdslabs(parameters_copy)

            if self.calctype == 'bulk':
                from ..structure_generators import GenerateBulk
                return GenerateBulk({'bulk': self.parameters['bulk']})
            if self.calctype == 'gas':
                from ..structure_generators import GenerateGas
                return GenerateGas({'gas': self.parameters['gas']})

    def run(self):
        # If there are matching entries, this is easy, just dump the matching entries
        # into a pickle file
        if len(self.matching_doc) > 0:
            with self.output().temporary_path() as self.temp_output_path:
                pickle.dump(self.matching_doc, open(self.temp_output_path, 'wb'))

        # Otherwise, we're missing a structure, so we need to submit whatever the
        # requirement returned
        else:
            launchpad = fwhs.get_launchpad()
            tosubmit = []

            # A way to append `tosubmit`, but specialized for gas relaxations
            if self.calctype == 'gas':
                name = {'vasp_settings': utils.unfreeze_dict(self.parameters['gas']['vasp_settings']),
                        'gasname': self.parameters['gas']['gasname'],
                        'calculation_type': 'gas phase optimization'}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    atoms = make_atoms_from_doc(pickle.load(open(self.input().fn, 'rb'))[0])
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       utils.unfreeze_dict(self.parameters['gas']['vasp_settings']),
                                                       max_atoms=defaults.BULK_SETTINGS['max_atoms']))

            # A way to append `tosubmit`, but specialized for bulk relaxations
            if self.calctype == 'bulk':
                name = {'vasp_settings': utils.unfreeze_dict(self.parameters['bulk']['vasp_settings']),
                        'mpid': self.parameters['bulk']['mpid'],
                        'calculation_type': 'unit cell optimization'}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    atoms = make_atoms_from_doc(pickle.load(open(self.input().fn, 'rb'))[0])
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       utils.unfreeze_dict(self.parameters['bulk']['vasp_settings']),
                                                       max_atoms=self.parameters['bulk']['max_atoms']))

            # A way to append `tosubmit`, but specialized for slab relaxations
            if self.calctype == 'slab':
                slab_docs = pickle.load(open(self.input()[0].fn, 'rb'))
                atoms_list = [make_atoms_from_doc(slab_doc) for slab_doc in slab_docs
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
                        'vasp_settings': utils.unfreeze_dict(self.parameters['slab']['vasp_settings']),
                        'calculation_type': 'slab optimization',
                        'num_slab_atoms': len(atoms)}
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       utils.unfreeze_dict(self.parameters['slab']['vasp_settings']),
                                                       max_atoms=self.parameters['bulk']['max_atoms'],
                                                       max_miller=self.parameters['slab']['max_miller']))

            # A way to append `tosubmit`, but specialized for surface energy calculations. Pretty much
            # identical to slab calcs, but different calculation type to keep them seperate and using
            # symmetric slab constraints (keep top and bottom layers free. For that reason, there is
            # thus not top
            if self.calctype == 'slab_surface_energy':
                slab_docs = pickle.load(open(self.input()[0].fn, 'rb'))
                atoms_list = [make_atoms_from_doc(slab_doc) for slab_doc in slab_docs
                              if float(np.round(slab_doc['tags']['shift'], 2)) ==
                              float(np.round(self.parameters['slab']['shift'], 2))]
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
                        'vasp_settings': utils.unfreeze_dict(self.parameters['slab']['vasp_settings']),
                        'calculation_type': 'slab_surface_energy optimization',
                        'num_slab_atoms': len(atoms)}
                atoms.constraints = []
                atoms = utils.constrain_slab(atoms, symmetric=True)
                if len(fwhs.running_fireworks(name, launchpad)) == 0:
                    tosubmit.append(fwhs.make_firework(atoms, name,
                                                       utils.unfreeze_dict(self.parameters['slab']['vasp_settings']),
                                                       max_atoms=self.parameters['bulk']['max_atoms'],
                                                       max_miller=self.parameters['slab']['max_miller']))

            # A way to append `tosubmit`, but specialized for adslab relaxations
            if self.calctype == 'slab+adsorbate':
                fpd_structs = pickle.load(open(self.input().fn, 'rb'))

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


                #Now that we use the relaxed bulk catalog, multiple adslabs might come back with different
                # shifts. We didn't have to check this before because the results were only
                # for a very specific relaxed surface
                if not np.isnan(self.parameters['slab']['shift']):
                    matching_docs = [doc for doc in fpd_structs if
                                     np.abs(doc['shift']-self.parameters['slab']['shift']) < 1e-4 and
                                     doc['top'] == self.parameters['slab']['top']]
                else:
                    matching_docs = fpd_structs

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
                            'vasp_settings': utils.unfreeze_dict(self.parameters['adsorption']['vasp_settings']),
                            'num_slab_atoms': self.parameters['adsorption']['num_slab_atoms'],
                            'slabrepeat': self.parameters['adsorption']['slabrepeat'],
                            'calculation_type': 'slab+adsorbate optimization'}
                    if 'adsorbate_rotation' in self.parameters['adsorption']['adsorbates'][0]:
                        name['adsorbate_rotation'] = utils.unfreeze_dict(self.parameters['adsorption']['adsorbates'][0]['adsorbate_rotation'])
                    # If there is no adsorbate, then the 'adsorption_site' key is irrelevant, as is the rotation
                    if name['adsorbate'] == '':
                        del name['adsorption_site']

                    atoms = doc['atoms']

                    # Add the firework if it's not already running
                    if len(fwhs.running_fireworks(name, launchpad)) == 0:
                        tosubmit.append(fwhs.make_firework(atoms, name,
                                                           utils.unfreeze_dict(self.parameters['adsorption']['vasp_settings']),
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
                raise RuntimeError('SubmitToFW unable to complete, waiting on a FW')
            else:
                raise RuntimeError('SubmitToFW unable to complete because there is nothing to submit')

    def output(self):
        return luigi.LocalTarget(GASDB_PATH+'/pickles/%s/%s.pkl' % (type(self).__name__, self.task_id))
