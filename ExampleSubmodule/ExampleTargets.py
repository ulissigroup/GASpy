'''
This set of classes are examples of calculations that you could submit to Luigi. Reference the
"RunExampleTarget.sh" file for the command line syntax used to submit the job.

Note that some of these classes require the calling of *.db files. We recommend storing these
*.db files in the GASpy/ directory and then putting your modified version of "ExampleTargets.py"
file into a submodule of GASpy (e.g., GASpy_regressions). If you do this, ensure that when you
ase.db.connect(*.db) in this file that you use ase.db.connect(../*.db) to fetch the *.db in the
GASpy/ directory, not the submodule directory.
'''


DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB/'


# Add the parent directory (i.e., GASpy) to the PYTHONPATH so that we can import the GASpy module
import sys
sys.path.append("..")
from collections import OrderedDict
import random
import cPickle as pickle
# from multiprocessing import Pool
from gaspy_toolbox import DumpSitesLocalDB
from gaspy_toolbox import FingerprintGeneratedStructures
from gaspy_toolbox import default_parameter_bulk
from gaspy_toolbox import default_parameter_gas
from gaspy_toolbox import default_parameter_slab
from gaspy_toolbox import default_parameter_adsorption
from gaspy_toolbox import DumpToLocalDB
from gaspy_toolbox import CalculateEnergy
from gaspy_toolbox import UpdateAllDB
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.matproj.rest import MPRester
import luigi
from ase.db import connect
import numpy as np
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.linear_model import LinearRegression
# from scipy.sparse import coo_matrix


class UpdateDBs(luigi.WrapperTask):
    """
    This class calls on the DumpToAuxDB class in gaspy_toolbox.py so that we can
    dump the fireworks database into the quick-access-mlab database. We would normally
    just use fireworks, but calling fireworks from a remote cluster is slow. So we speed
    up the calls by dumping the data to mlab, where querying is fast.
    """
    def requires(self):
        """
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        """
        yield UpdateAllDB()


class ExampleSingleSiteSubmission(luigi.WrapperTask):
    def requires(self):
        """
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        """
        # Find and submit CO at the 4-fold Ni site on the top side of mp-23(100)
        ads_parameter = default_parameter_adsorption('CO')
        # When there is a 'fp' directionary in parameters['adsorption']['adsorbates'][0],
        # it indicatees that we want to place an adsorbate at a type of site instead of an
        # exact XYZ location.  In that case, it will find the appropriate slab, find all
        # adsorbate positions, fingerprint them, and use those that match every every key
        # in the 'fp' dictionary.
        ads_parameter['adsorbates'][0]['fp'] = {'coordination':'Ni-Ni-Ni-Ni'}
        parameters = {'bulk': default_parameter_bulk('mp-23'),
                      'slab':default_parameter_slab([1,0,0], True, 0),
                      'gas':default_parameter_gas('CO'),
                      'adsorption':ads_parameter}
        yield DumpToLocalDB(parameters=parameters)

        # Find and submit H at a 3-fold Ni site on the top side of mp-23(111)
        ads_parameter = default_parameter_adsorption('H')
        ads_parameter['adsorbates'][0]['fp'] = {'coordination':'Ni-Ni-Ni'}
        parameters = {"bulk":default_parameter_bulk('mp-23'),
                      'slab':default_parameter_slab([1,1,1], True, 0),
                      'gas':default_parameter_gas('CO'),
                      'adsorption':ads_parameter}
        yield DumpToLocalDB(parameters=parameters)

        # Find and submit CO at every Ni-Ni bridge site on the top side of mp-23(100)
        ads_parameter = default_parameter_adsorption('CO')
        ads_parameter['adsorbates'][0]['fp'] = {'coordination':'Ni-Ni'}
        ads_parameter['numtosubmit'] = -1 # Submit all matching sites
        parameters = {"bulk":default_parameter_bulk('mp-23'),
                      'slab':default_parameter_slab([1,0,0], True, 0),
                      'gas':default_parameter_gas('CO'),
                      'adsorption':ads_parameter}
        yield DumpToLocalDB(parameters=parameters)


class StudyCoordinationSites(luigi.WrapperTask):
    """
    This class is meant to be called by Luigi to begin relaxations of a particular set of
    adsorption sites.
    """
    def requires(self):
        """
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        """

        # Get all of the enumerated configurations
        with connect(DB_LOC+'/enumerated_adsorption_sites.db') as con:
            rows = [row for row in con.select()]

        # Get all of the adsorption energies we've already calculated
        with connect(DB_LOC+'/adsorption_energy_database.db') as con:
            resultRows = [row for row in con.select()]

        # Find all of the unique sites at the first level of coordination for enumerated configurations
        unique_coord, inverse = np.unique([str([row.coordination]) for row in rows], return_inverse=True)
        random.seed(42)

        print 'Number of unique sites to first order: %d'%len(unique_coord)


        # select a number of random site types to investigate
        inds = range(len(unique_coord))
        random.shuffle(inds)
        ind_to_run = inds

        # Add on-top configuration for each atom type
        for ind in inds:
            if len(unique_coord[ind].split('-')) <= 1 and ind not in ind_to_run:
                ind_to_run += [ind]

        # For each configuration, submit
        for ind in ind_to_run:
            print('Let\'s try %s!'%unique_coord[ind])
            indices, natoms = zip(*[[i, rows[i].natoms] for i in range(len(inverse)) if inverse[i] == ind])
            rowind = indices[np.argmin(natoms)]
            row = rows[rowind]
            print(row.miller)
            print(row.mpid)
            for adsorbate in ['CO']:
                if len([result for result in resultRows
                        if result.adsorbate == adsorbate
                        and (result.coordination == row.coordination
                             or result.initial_coordination == row.coordination) 
                       ]
                      ) <=1:
                    ads_parameter = default_parameter_adsorption(adsorbate)
                    ads_parameter['adsorbates'][0]['fp'] = {'coordination':row.coordination}
                    parameters = {"bulk": default_parameter_bulk(row.mpid),
                                  'slab':default_parameter_slab(list(eval(row.miller)), row.top, row.shift),
                                  'gas':default_parameter_gas('CO'),
                                  'adsorption':ads_parameter}
                    yield CalculateEnergy(parameters=parameters)


class EnumerateAlloys(luigi.WrapperTask):
    """
    This class is meant to be called by Luigi to begin relaxations of a database of alloys
    """
    max_index = luigi.IntParameter(2)
    writeDB = luigi.BoolParameter(False)
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

        whitelist = ['Pt', 'Ag', 'Cu', 'Pd', 'Ni', 'Au', 'Ga', 'Rh', 'Re',
                     'W', 'Al', 'Co', 'H', 'N', 'Ir', 'In']

        whitelist = ['Pd', 'Cu', 'Au', 'Ag', 'Pt', 'Rh', 'Re', 'Ni', 'Co',
                     'Ir', 'W', 'Al', 'Ga', 'In', 'H', 'N', 'Os', 'Fe']
        # whitelist=['Pd','Cu','Au','Ag']
        restricted_elements = [el for el in all_elements if el not in whitelist]

        # Query MP for all alloys that are stable, near the lower hull, and don't have one of the restricted elements
        with MPRester("MGOdX3P4nI18eKvE") as m:
            results = m.query({"elements":{"$nin": restricted_elements},
                               "e_above_hull":{"$lt":0.1},
                               "formation_energy_per_atom":{"$lte":0.0}},
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
        all_miller = map(processStruc, results)

        for facets in all_miller:
            for facet in facets:
                if self.writeDB:
                    yield DumpSitesLocalDB(parameters=OrderedDict(unrelaxed=True,
                                                                  bulk=default_parameter_bulk(facet[0]),
                                                                  slab=default_parameter_slab(facet[1], True, 0),
                                                                  gas=default_parameter_gas('CO'),
                                                                  adsorption=default_parameter_adsorption('U',
                                                                                                           "[  3.36   1.16  24.52]",
                                                                                                           "(1, 1)", 24)
                                                                    )
                                             )
                else:
                    yield FingerprintGeneratedStructures(parameters=OrderedDict(unrelaxed=True,
                                                                                bulk=default_parameter_bulk(facet[0]),
                                                                                slab=default_parameter_slab(facet[1], True, 0),
                                                                                gas=default_parameter_gas('CO'),
                                                                                adsorption=default_parameter_adsorption('U',
                                                                                                                      "[  3.36   1.16  24.52]",
                                                                                                                      "(1, 1)", 24)))


class PredictAndSubmit(luigi.WrapperTask):
    """
    This is meant to be called by Luigi to begin relaxations of slab+adsorbate systems
    whose energies we have predicted (via regression) but not yet calculated via DFT.
    See the `primary_coordination_prediction_next.py` file in the GASpy_regressions
    submodule for details regarding how we created the *.pkl file referenced here.
    """
    def requires(self):
        """
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        """
        # Get all of the adsorption sites we've already identified
        with connect(DB_LOC+'/enumerated_adsorption_sites.db') as conEnum:
            adsorption_rows_catalog = [row for row in conEnum.select()]

        # Get all of the adsorption energies we've already calculated
        with connect(DB_LOC+'/adsorption_energy_database.db') as con:
            resultRows = [row for row in con.select()]

        # Load the regression's predictions from a pickle. You may need to change the
        # location depending on your folder structure.
        dEprediction = pickle.load(open('../GASpy_regressions/primary_coordination_prediction.pkl'))
        matching = []
        for dE, row in zip(dEprediction['CO'], adsorption_rows_catalog):
            if (dE > -0.7
                    and dE < -0.4
                    and 'Cu' not in row.formula
                    and 'Al' not in row.formula
                    and np.max(eval(row.miller)) <= 2
                    and row.natoms < 40
                    and len([row2 for row2 in resultRows
                             if row2.adsorbate == 'CO'
                             and ((row2.coordination == row.coordination
                                   and row2.nextnearestcoordination == row.nextnearestcoordination)
                                  or (row2.initial_coordination == row.coordination
                                      and row2.initial_nextnearestcoordination == row.nextnearestcoordination))
                            ]) == 0):
                matching.append([dE, row])

        ncoord, ncoord_index = np.unique([str([row[1].coordination, row[1].nextnearestcoordination])
                                          for row in matching], return_index=True)
        ncoord_index = sorted(ncoord_index, key=lambda x: np.abs(-0.55-matching[x][0]))

        #ncoord, ncoord_index = np.unique([str([row[1].coordination])
        #                                  for row in matching], return_index=True)

        print len(ncoord_index)
        ncoord_index = ncoord_index[0:100]
        # Initiate the DFT relaxations/calculations of these systems
        for ind in ncoord_index:
            row = matching[ind][1]
            ads_parameter = default_parameter_adsorption('CO')
            ads_parameter['adsorbates'][0]['fp'] = {'coordination':row.coordination,
                                                    'nextnearestcoordination':row.nextnearestcoordination}
            parameters = {'bulk': default_parameter_bulk(row.mpid),
                          'slab':default_parameter_slab(list(eval(row.miller)), row.top, row.shift),
                          'gas':default_parameter_gas('CO'),
                          'adsorption':ads_parameter}
            yield CalculateEnergy(parameters=parameters)
