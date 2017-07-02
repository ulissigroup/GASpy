from collections import OrderedDict
import luigi
from ..utils import get_aux_db


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
#         def logical_fun(row, search):
#             '''
#             This function compares a search dictionary with another dictionary row,
#             and returns true if all entries in search match the corresponding entry in row
#             '''
#             rowdict = row.__dict__
#             for key in search:
#                 if key not in rowdict:
#                     return False
#                 elif rowdict[key] != search[key]:
#                     return False
#             return True

        # Define a dictionary that will be used to search the Auxiliary database and find
        # the correct entry
        if self.calctype == 'gas':
            search_strings = {'type':'gas',
                              'fwname.gasname':self.parameters['gas']['gasname']}
            for key in self.parameters['gas']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s'%key] = \
                        self.parameters['gas']['vasp_settings'][key]
        elif self.calctype == 'slab':
            search_strings = {'type': 'slab',
                              'fwname.miller': list(self.parameters['slab']['miller']),
                              'fwname.top': self.parameters['slab']['top'],
                              'fwname.shift': self.parameters['slab']['shift'],
                              'fwname.mpid': self.parameters['bulk']['mpid']}
            for key in self.parameters['slab']['vasp_settings']:
                if key not in ['isym']:
                    search_strings['fwname.vasp_settings.%s'%key] = \
                            self.parameters['slab']['vasp_settings'][key]
        elif self.calctype == 'bulk':
            search_strings = {'type':'bulk',
                              'fwname.mpid':self.parameters['bulk']['mpid']}
            for key in self.parameters['bulk']['vasp_settings']:
                search_strings['fwname.vasp_settings.%s'%key] = \
                        self.parameters['bulk']['vasp_settings'][key]
        elif self.calctype == 'slab+adsorbate':
            search_strings = {'type':'slab+adsorbate',
                              'fwname.miller':list(self.parameters['slab']['miller']),
                              'fwname.top':self.parameters['slab']['top'],
                              'fwname.shift':self.parameters['slab']['shift'],
                              'fwname.mpid':self.parameters['bulk']['mpid'],
                              'fwname.adsorbate':self.parameters['adsorption']['adsorbates'][0]['name']}
            for key in self.parameters['adsorption']['vasp_settings']:
                if key not in ['nsw', 'isym']:
                    search_strings['fwname.vasp_settings.%s'%key] = \
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
        with get_aux_db() as aux_db:
            self.matching_row = list(aux_db.find(search_strings))
        #print('Search string:  %s', % search_strings)

        # If there are no matching entries, we need to yield a requirement that will
        # generate the necessary unrelaxed structure
        if len(self.matching_row) == 0:
            if self.calctype == 'slab':
                return [GenerateSurfaces(OrderedDict(bulk=self.parameters['bulk'],
                                                     slab=self.parameters['slab'])),
                        GenerateSurfaces(OrderedDict(unrelaxed=True,
                                                     bulk=self.parameters['bulk'],
                                                     slab=self.parameters['slab']))]
            if self.calctype == 'slab+adsorbate':
                # Return the base structure, and all possible matching ones for the surface
                search_strings = {'type':'slab+adsorbate',
                                  'fwname.miller':list(self.parameters['slab']['miller']),
                                  'fwname.top':self.parameters['slab']['top'],
                                  'fwname.mpid':self.parameters['bulk']['mpid'],
                                  'fwname.adsorbate':self.parameters['adsorption']['adsorbates'][0]['name']}
                with get_aux_db() as aux_db:
                    self.matching_rows_all_calcs = list(aux_db.find(search_strings))
                return FingerprintUnrelaxedAdslabs(self.parameters)
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

            # A way to append `tosubmit`, but specialized for bulk relaxations
            if self.calctype == 'bulk':
                name = {'vasp_settings':self.parameters['bulk']['vasp_settings'],
                        'mpid':self.parameters['bulk']['mpid'],
                        'calculation_type':'unit cell optimization'}
                if len(running_fireworks(name, launchpad)) == 0:
                    atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
                    tosubmit.append(make_firework(atoms, name,
                                                  self.parameters['bulk']['vasp_settings'],
                                                  max_atoms=self.parameters['bulk']['max_atoms']))

            # A way to append `tosubmit`, but specialized for gas relaxations
            if self.calctype == 'gas':
                name = {'vasp_settings':self.parameters['gas']['vasp_settings'],
                        'gasname':self.parameters['gas']['gasname'],
                        'calculation_type':'gas phase optimization'}
                if len(running_fireworks(name, launchpad)) == 0:
                    atoms = mongo_doc_atoms(pickle.load(self.input().open())[0])
                    tosubmit.append(make_firework(atoms, name,
                                                  self.parameters['gas']['vasp_settings']))

            # A way to append `tosubmit`, but specialized for slab relaxations
            if self.calctype == 'slab':
                slab_list = pickle.load(self.input()[0].open())
                '''
                An earlier version of GASpy did not correctly log the slab shift, and so many
                of our calculations/results do no have shifts. If there is either zero or more
                one shift value, then this next paragraph (i.e., all of the following code
                before the next blank line) deals with it in a "hacky" way.
                '''
                atomlist = [mongo_doc_atoms(slab) for slab in slab_list
                            if float(np.round(slab['tags']['shift'], 2)) == float(np.round(self.parameters['slab']['shift'], 2))
                            and slab['tags']['top'] == self.parameters['slab']['top']]
                if len(atomlist) > 1:
                    print('We have more than one slab system with identical shifts:\n%s' \
                          % self.input()[0].fn)
                elif len(atomlist) == 0:
                    # We couldn't find the desired shift value in the surfaces
                    # generated for the relaxed bulk, so we need to try to find
                    # it by comparison with the reference (unrelaxed) surfaces
                    slab_list_unrelaxed = pickle.load(self.input()[1].open())
                    atomlist_unrelaxed = [mongo_doc_atoms(slab) for slab in slab_list_unrelaxed
                                          if float(np.round(slab['tags']['shift'], 2)) == \
                                             float(np.round(self.parameters['slab']['shift'], 2))
                                          and slab['tags']['top'] == self.parameters['slab']['top']]
                    #if len(atomlist_unrelaxed)==0:
                    #    pprint(slab_list_unrelaxed)
                    #    pprint('desired shift: %1.4f'%float(np.round(self.parameters['slab']['shift'],2)))
                    # we need all of the relaxed slabs in atoms form:
                    all_relaxed_surfaces = [mongo_doc_atoms(slab) for slab in slab_list
                                            if slab['tags']['top'] == self.parameters['slab']['top']]
                    # We use the average coordination as a descriptor of the structure,
                    # there should be a pretty large change with different shifts
                    def getCoord(x):
                        return average_coordination_number([AseAtomsAdaptor.get_structure(x)])
                    # Get the coordination for the unrelaxed surface w/ correct shift
                    reference_coord = getCoord(atomlist_unrelaxed[0])
                    # get the coordination for each relaxed surface
                    relaxed_coord = map(getCoord, all_relaxed_surfaces)
                    # We want to minimize the distance in these dictionaries
                    def getDist(x, y):
                        vals = []
                        for key in x:
                            vals.append(x[key]-y[key])
                        return np.linalg.norm(vals)
                    # Get the distances to the reference coordinations
                    dist = map(lambda x: getDist(x, reference_coord), relaxed_coord)
                    # Grab the atoms object that minimized this distance
                    atoms = all_relaxed_surfaces[np.argmin(dist)]
                    print('Unable to find a slab with the correct shift, but found one with max \
                          position difference of %1.4f!'%np.min(dist))

                # If there is a shift value in the results, then continue as normal.
                elif len(atomlist) == 1:
                    atoms = atomlist[0]
                name = {'shift':self.parameters['slab']['shift'],
                        'mpid':self.parameters['bulk']['mpid'],
                        'miller':self.parameters['slab']['miller'],
                        'top':self.parameters['slab']['top'],
                        'vasp_settings':self.parameters['slab']['vasp_settings'],
                        'calculation_type':'slab optimization',
                        'num_slab_atoms':len(atoms)}
                #print(name)
                if len(running_fireworks(name, launchpad)) == 0:
                    tosubmit.append(make_firework(atoms, name,
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
                    matching_rows = [row for row in fpd_structs
                                     if matchFP(row, self.parameters['adsorption']['adsorbates'][0]['fp'])]
                else:
                    if self.parameters['adsorption']['adsorbates'][0]['name'] != '':
                        matching_rows = [row for row in fpd_structs
                                         if row['adsorption_site'] == \
                                            self.parameters['adsorption']['adsorbates'][0]['adsorption_site']]
                    else:
                        matching_rows = [row for row in fpd_structs]
                #if len(matching_rows) == 0:
                    #print('No rows matching the desired FP/Site!')
                    #print('Desired sites:')
                    #pprint(str(self.parameters['adsorption']['adsorbates'][0]['fp']))
                    #print('Available Sites:')
                    #pprint(fpd_structs)
                    #pprint(self.input().fn)
                    #pprint(self.parameters)

                # If there is no adsorbate, then trim the matching_rows to the first row we found.
                # Otherwise, trim the matching_rows to `numtosubmit`, a user-specified value that
                # decides the maximum number of fireworks that we want to submit.
                if self.parameters['adsorption']['adsorbates'][0]['name'] == '':
                    matching_rows = matching_rows[0:1]
                elif 'numtosubmit' in self.parameters['adsorption']:
                    matching_rows = matching_rows[0:self.parameters['adsorption']['numtosubmit']]

                # Add each of the matchig rows to `tosubmit`
                for row in matching_rows:
                    # The name of our firework is actually a dictionary, as defined here
                    name = {'mpid':self.parameters['bulk']['mpid'],
                            'miller':self.parameters['slab']['miller'],
                            'top':self.parameters['slab']['top'],
                            'shift':row['shift'],
                            'adsorbate':self.parameters['adsorption']['adsorbates'][0]['name'],
                            'adsorption_site':row['adsorption_site'],
                            'vasp_settings':self.parameters['adsorption']['vasp_settings'],
                            'num_slab_atoms':self.parameters['adsorption']['num_slab_atoms'],
                            'slabrepeat':self.parameters['adsorption']['slabrepeat'],
                            'calculation_type':'slab+adsorbate optimization'}
                    # If there is no adsorbate, then the 'adsorption_site' key is irrelevant
                    if name['adsorbate'] == '':
                        del name['adsorption_site']

                    '''
                    This next paragraph (i.e., code until the next blank line) is a prototyping
                    skeleton for GASpy Issue #14
                    '''
                    # First, let's see if we can find a reasonable guess for the row:
                    # guess_rows=[row2 for row2 in self.matching_rows_all_calcs if matchFP(fingerprint(row2['atoms'], ), row)]
                    guess_rows = []
                    # We've found another calculation with exactly the same fingerprint
                    if len(guess_rows) > 0:
                        guess = guess_rows[0]
                        # Get the initial adsorption site of the identified row
                        ads_site = np.array(map(eval, guess['fwname']['adsorption_site'].strip().split()[1:4]))
                        atoms = row['atoms']
                        atomsguess = guess['atoms']
                        # For each adsorbate atom, move it the same relative amount as in the guessed configuration
                        lenAdsorbates = len(Atoms(self.parameters['adsorption']['adsorbates'][0]['name']))
                        for ind in range(-lenAdsorbates, len(atoms)):
                            atoms[ind].position += atomsguess[ind].position-ads_site
                    else:
                        atoms = row['atoms']
                    if len(guess_rows) > 0:
                        name['guessed_from'] = {'xc':guess['fwname']['vasp_settings']['xc'],
                                                'encut':guess['fwname']['vasp_settings']['encut']}

                    # Add the firework if it's not already running
                    if len(running_fireworks(name, launchpad)) == 0:
                        tosubmit.append(make_firework(atoms, name,
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
                print('Just submitted the following Fireworks:')
                for i, submit in enumerate(tosubmit):
                    print('    Submission number %s' % i)
                    print_dict(submit, indent=2)

    def output(self):
        return luigi.LocalTarget(LOCAL_DB_PATH+'/pickles/%s.pkl'%(self.task_id))
