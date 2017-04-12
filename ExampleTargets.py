from collections import OrderedDict
import random
from generate_database_luigi import WriteConfigsLocalDB
from generate_database_luigi import FingerprintGeneratedStructures
from generate_database_luigi import default_parameter_bulk
from generate_database_luigi import default_parameter_gas
from generate_database_luigi import default_parameter_slab
from generate_database_luigi import default_parameter_adsorption
from generate_database_luigi import WriteAdsorptionConfig
from generate_database_luigi import CalculateEnergy
from generate_database_luigi import get_db
from generate_database_luigi import UpdateDB
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.matproj.rest import MPRester
import luigi
from ase.db import connect
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from scipy.sparse import coo_matrix
from multiprocessing import Pool
import cPickle as pickle



class ExampleSingleSiteSubmission(luigi.WrapperTask):
    def requires(self):
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
        yield WriteAdsorptionConfig(parameters=parameters)

        # Find and submit H at a 3-fold Ni site on the top side of mp-23(111)
        ads_parameter = default_parameter_adsorption('H')
        ads_parameter['adsorbates'][0]['fp'] = {'coordination':'Ni-Ni-Ni'}
        parameters = {"bulk":default_parameter_bulk('mp-23'),
                      'slab':default_parameter_slab([1,1,1], True, 0),
                      'gas':default_parameter_gas('CO'),
                      'adsorption':ads_parameter}
        yield WriteAdsorptionConfig(parameters=parameters)

        # Find and submit CO at every Ni-Ni bridge site on the top side of mp-23(100)
        ads_parameter = default_parameter_adsorption('CO')
        ads_parameter['adsorbates'][0]['fp'] = {'coordination':'Ni-Ni'}
        ads_parameter['numtosubmit'] = -1 # Submit all matching sites
        parameters = {"bulk":default_parameter_bulk('mp-23'),
                      'slab':default_parameter_slab([1,0,0], True, 0),
                      'gas':default_parameter_gas('CO'),
                      'adsorption':ads_parameter}
        yield WriteAdsorptionConfig(parameters=parameters)


class UpdateDFTAdsorptionEnergies(luigi.WrapperTask):
    def requires(self):
        yield UpdateDB()

        # Get every row in the mongo database of completed fireworks results
        relaxed_rows = get_db().find({'type':'slab+adsorbate'})

        # Find unique adsorption sites (in case multiple rows are basically the same adsorbate/position/etc
        unique_configs = np.unique([str([row['fwname']['mpid'],
                                         row['fwname']['miller'],
                                         row['fwname']['top'],
                                         row['fwname']['adsorption_site'],
                                         row['fwname']['adsorbate'],
                                         row['fwname']['num_slab_atoms'],
                                         row['fwname']['slabrepeat'],
                                         row['fwname']['shift']])
                                    for row in relaxed_rows
                                    if row['fwname']['adsorbate'] != ''])

        # For each adsorbate/configuration, make a task to write the results to the output database
        for row in unique_configs:
            mpid, miller, top, adsorption_site, adsorbate, num_slab_atoms, slabrepeat, shift = eval(row)
            parameters = {'bulk':default_parameter_bulk(mpid),
                          'gas':default_parameter_gas(gasname='CO'),
                          'slab':default_parameter_slab(miller=miller, shift=shift, top=top),
                          'adsorption':default_parameter_adsorption(adsorbate=adsorbate,
                                                                  num_slab_atoms=num_slab_atoms,
                                                                  slabrepeat=slabrepeat,
                                                                  adsorption_site=adsorption_site)
                         }
            yield WriteAdsorptionConfig(parameters)



class StudyCoordinationSites(luigi.WrapperTask):
    def requires(self):

        # Get all of the enumerated configurations
        with connect('enumerated_adsorption_sites.db') as con:
            rows = [row for row in con.select()]

        # Get all of the adsorption energies we've already calculated
        with connect('adsorption_energy_database.db') as con:
            resultRows = [row for row in con.select()]

        # Find all of the unique sites at the first level of coordination for results
        unique_result_coordination = np.unique([row.coordination for row in resultRows])

        # Find all of the unique sites at the first level of coordination for enumerated configurations
        unique_coord, inverse = np.unique([str([row.coordination]) for row in rows], return_inverse=True)
        random.seed(42)

        print('Number of unique sites to first order: %d'%len(unique_coord))


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
    max_index = luigi.IntParameter(2)
    writeDB = luigi.BoolParameter(False)
    def requires(self):
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
            # results = m.query({"elements":{"$in": ['Cu','Pd','Au']},"e_above_hull":{"$lt":0.1},"formation_energy_per_atom":{"$lt":0.0}},["pretty_formula","formula",'spacegroup','material id','taskid','task_id'],mp_decode=False)

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
                    yield WriteConfigsLocalDB(parameters=OrderedDict(unrelaxed=True,
                                                                     bulk=default_parameter_bulk(facet[0]),
                                                                     slab=default_parameter_slab(facet[1], True, 0),
                                                                     gas=default_parameter_gas('CO'),
                                                                     adsorption=default_parameter_adsorption('U',
                                                                                                           "[  3.36   1.16  24.52]",
                                                                                                           "(1, 1)",24)
                                                                    )
                                             )
                else:
                    yield FingerprintGeneratedStructures(parameters=OrderedDict(unrelaxed=True,
                                                                                bulk=default_parameter_bulk(facet[0]),
                                                                                slab=default_parameter_slab(facet[1], True, 0),
                                                                                gas=default_parameter_gas('CO'),
                                                                                adsorption=default_parameter_adsorption('U',
                                                                                                                      "[  3.36   1.16  24.52]",
                                                                                                                      "(1, 1)",24)))

class PredictAndSubmit(luigi.WrapperTask):
    def requires(self):
        conEnum=connect('../GASpy/enumerated_adsorption_sites.db')
        adsorption_rows_catalog=[row for row in conEnum.select()]

        # Get all of the adsorption energies we've already calculated
        with connect('adsorption_energy_database.db') as con:
            resultRows = [row for row in con.select()]

        dEprediction=pickle.load(open('../GASpy_regressions/primary_coordination_prediction.pkl'))
        matching=[]
        for dE,row in zip(dEprediction['CO'],adsorption_rows_catalog):
            if dE>-1.2 and dE<-0.5 and 'Al' not in row.formula and 'Cu' not in row.formula and row.natoms<40 and len([row2 for row2 in resultRows if row2.coordination==row.coordination and row2.nextnearestcoordination==row.nextnearestcoordination])>0:
                matching.append([dE,row])

        ncoord,ncoord_index=np.unique([str([row[1].coordination,row[1].nextnearestcoordination]) for row in matching],return_index=True)

        for ind in ncoord_index:
            row=matching[ind][1]
            ads_parameter = default_parameter_adsorption('CO')
            ads_parameter['adsorbates'][0]['fp'] = {'coordination':row.coordination,'nextnearestcoordination':row.nextnearestcoordination}
            parameters = {'bulk': default_parameter_bulk(row.mpid),
                          'slab':default_parameter_slab(list(eval(row.miller)), row.top, row.shift),
                          'gas':default_parameter_gas('CO'),
                          'adsorption':ads_parameter}
            yield CalculateEnergy(parameters=parameters)
            #parameterRPBE=copy.deepcopy(parameters)
            #parameterRPBE['bulk']['vasp_settings']['xc']='rpbe'
            #parameterRPBE['slab']['vasp_settings']['xc']='rpbe'
            #parameterRPBE['gas']['vasp_settings']['xc']='rpbe'
            #parameterRPBE['asdsorption']['vasp_settings']['xc']='rpbe'
            #yield CalculateEnergy(


