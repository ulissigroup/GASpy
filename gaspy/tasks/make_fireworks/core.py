'''
This module houses the functions needed to make and submit FireWorks rockets
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
import luigi
from ..atoms_generators import GenerateGas, GenerateBulk
from ... import defaults
from ...mongo import make_atoms_from_doc
from ...utils import unfreeze_dict
from ...fireworks_helper_scripts import make_firework, submit_fwork

GAS_SETTINGS = defaults.GAS_SETTINGS
BULK_SETTINGS = defaults.BULK_SETTINGS


class MakeGasFW(luigi.Task):
    '''
    This task will create and submit a gas relaxation for you.

    Args:
        gas_name        A string indicating which gas you want to relax
        vasp_settings   A dictionary containing your VASP settings
    '''
    gas_name = luigi.Parameter()
    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])

    def requires(self):
        return GenerateGas(gas_name=self.gas_name)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        vasp_settings = unfreeze_dict(self.vasp_settings)
        fw_name = {'gasname': self.gas_name,
                   'vasp_settings': vasp_settings,
                   'calculation_type': 'gas phase optimization'}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              vasp_settings=vasp_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeBulkFW(luigi.Task):
    '''
    This task will create and submit a bulk relaxation for you.

    Args:
        mpid            A string indicating the mpid of the bulk
        vasp_settings   A dictionary containing your VASP settings
    '''
    mpid = luigi.Parameter()
    vasp_settings = luigi.DictParameter(BULK_SETTINGS['vasp'])

    def requires(self):
        return GenerateBulk(mpid=self.mpid)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        vasp_settings = unfreeze_dict(self.vasp_settings)
        fw_name = {'mpid': self.mpid,
                   'vasp_settings': vasp_settings,
                   'calculation_type': 'unit cell optimization'}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              vasp_settings=vasp_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


#class MakeBulkFW(luigi.Task):
#    '''
#    This class accepts a luigi.Task (e.g., relax a structure), then checks to
#    see if this task is already logged in the Auxiliary vasp.mongo database. If
#    it is not, then it submits the task to our Primary FireWorks database.
#    '''
#    gas_name = luigi.Parameter()
#    vasp_settings = luigi.DictParameter(GAS_SETTINGS['vasp'])
#
#    def requires(self):
#        return GenerateGas(gas_name=self.gas_name)
#
#    def run(self, _test=False):
#        ''' Do not use `_test=True` unless you are unit testing '''
#        # Parse the input atoms object
#        with open(self.input().path, 'rb') as file_handle:
#            doc = pickle.load(file_handle)
#        atoms = make_atoms_from_doc(doc)
#
#        # Create and package the FireWork
#        fw_name = {'gasname': self.gas_name,
#                   'vasp_settings': unfreeze_dict(self.vasp_settings),
#                   'calculation_type': 'gas phase optimization'}
#        fwork = make_firework(atoms=atoms,
#                              fw_name=fw_name,
#                              vasp_settings=unfreeze_dict(self.vasp_settings))
#        wflow = Workflow([fwork], name='vasp optimization')
#
#        # Submit the FireWork to our launch pad
#        if _test is False:
#            lpad = get_launchpad()
#            lpad.add_wf(wflow)
#        # If we are unit testing, then DO NOT submit the FireWork
#        else:
#            return wflow
#
#        print('Just submitted the following Fireworks:')
#        print_dict(fwork.name, indent=1)
#
#    def output(self):
#        return make_task_output_object(self)
