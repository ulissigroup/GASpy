'''
This module houses the functions needed to make and submit FireWorks rockets.

Note that all of the tasks in this submodule will always show up as incomplete,
which means we will always allow users to make new FireWorks. It's on them to
not make redundant ones. If you want to make a FireWork only if it's not
already running or done, then you should use the
`gaspy.tasks.calculation_finders` submodule instead.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import pickle
import math
import numpy as np
import luigi
from .atoms_generators import GenerateGas, GenerateBulk, GenerateAdslabs
from .. import defaults
from ..mongo import make_atoms_from_doc
from ..utils import unfreeze_dict
from ..fireworks_helper_scripts import make_firework, submit_fwork, get_launchpad

DFT_CALCULATOR = defaults.DFT_CALCULATOR
GAS_SETTINGS = defaults.gas_settings()
BULK_SETTINGS = defaults.bulk_settings()
SLAB_SETTINGS = defaults.slab_settings()
ADSLAB_SETTINGS = defaults.adslab_settings()


class FireworkMaker(luigi.Task):
    _complete = False

    def complete(self):
        '''
        This task is designed to make and submit you a FireWork once every time
        you call it. To get Luigi to do this for us, it must be not be marked
        as "complete" until the `run` method changes the `self._complete` flag
        to `True`.
        '''
        return self._complete


class MakeGasFW(FireworkMaker):
    '''
    This task will create and submit a gas relaxation for you.

    Args:
        gas_name        A string indicating which gas you want to relax
        dft_settings    A dictionary containing your DFT settings
    '''
    gas_name = luigi.Parameter()
    dft_settings = luigi.DictParameter(GAS_SETTINGS[DFT_CALCULATOR])

    def requires(self):
        return GenerateGas(gas_name=self.gas_name)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'gas phase optimization',
                   'gasname': self.gas_name,
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Let Luigi know that we've made the FireWork
        self._complete = True

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeBulkFW(FireworkMaker):
    '''
    This task will create and submit a bulk relaxation for you.

    Args:
        mpid            A string indicating the mpid of the bulk
        dft_settings    A dictionary containing your DFT settings
    '''
    mpid = luigi.Parameter()
    dft_settings = luigi.DictParameter(BULK_SETTINGS[DFT_CALCULATOR])
    max_atoms = luigi.IntParameter(50)

    def requires(self):
        return GenerateBulk(mpid=self.mpid)

    def run(self, _testing=False):
        ''' Do not use `_test=True` unless you are unit testing '''
        # Parse the input atoms object
        with open(self.input().path, 'rb') as file_handle:
            doc = pickle.load(file_handle)
        atoms = make_atoms_from_doc(doc)

        # Don't make a bulk that is too big
        if len(atoms) > self.max_atoms:
            raise ValueError('The size of the bulk, %i, is larger than the '
                             'specified limit, %i. We will not be making this '
                             'bulk FireWork (%s)'
                             % (len(atoms), self.max_atoms, self.mpid))

        # Create, package, and submit the FireWork
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'unit cell optimization',
                   'mpid': self.mpid,
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Increase the priority because it's a bulk
        lpad = get_launchpad()
        lpad.set_priority(fwork.fw_id, 100)

        # Let Luigi know that we've made the FireWork
        self._complete = True

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeAdslabFW(FireworkMaker):
    '''
    This task will create and submit an adsorbate+slab (adslab) calculation

    Args:
        adsorption_site         A 3-tuple of floats containing the Cartesian
                                coordinates of the adsorption site you want to
                                make a FW for
        shift                   A float indicating the shift of the slab
        top                     A Boolean indicating whether the adsorption
                                site is on the top or the bottom of the slab
        dft_settings            A dictionary containing your DFT settings
                                for the adslab relaxation
        adsorbate_name          A string indicating which adsorbate to use. It
                                should be one of the keys within the
                                `gaspy.defaults.ADSORBATES` dictionary. If you
                                want an adsorbate that is not in the dictionary,
                                then you will need to add the adsorbate to that
                                dictionary.
        rotation                A dictionary containing the angles (in degrees)
                                in which to rotate the adsorbate after it is
                                placed at the adsorption site. The keys for
                                each of the angles are 'phi', 'theta', and
                                psi'.
        mpid                    A string indicating the Materials Project ID of
                                the bulk you want to enumerate sites from
        miller_indices          A 3-tuple containing the three Miller indices
                                of the slab[s] you want to enumerate sites from
        min_xy                  A float indicating the minimum width (in both
                                the x and y directions) of the slab (Angstroms)
                                before we enumerate adsorption sites on it.
        slab_generator_settings We use pymatgen's `SlabGenerator` class to
                                enumerate surfaces. You can feed the arguments
                                for that class here as a dictionary.
        get_slab_settings       We use the `get_slabs` method of pymatgen's
                                `SlabGenerator` class. You can feed the
                                arguments for the `get_slabs` method here
                                as a dictionary.
        bulk_dft_settings       A dictionary containing the DFT settings of
                                the relaxed bulk to enumerate slabs from
    '''
    adsorption_site = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    top = luigi.BoolParameter()
    dft_settings = luigi.DictParameter(ADSLAB_SETTINGS[DFT_CALCULATOR])

    # Passed to `GenerateAdslabs`
    adsorbate_name = luigi.Parameter()
    rotation = luigi.DictParameter(ADSLAB_SETTINGS['rotation'])
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    min_xy = luigi.FloatParameter(ADSLAB_SETTINGS['min_xy'])
    slab_generator_settings = luigi.DictParameter(SLAB_SETTINGS['slab_generator_settings'])
    get_slab_settings = luigi.DictParameter(SLAB_SETTINGS['get_slab_settings'])
    bulk_dft_settings = luigi.DictParameter(BULK_SETTINGS[DFT_CALCULATOR])

    def requires(self):
        return GenerateAdslabs(adsorbate_name=self.adsorbate_name,
                               rotation=self.rotation,
                               mpid=self.mpid,
                               miller_indices=self.miller_indices,
                               min_xy=self.min_xy,
                               slab_generator_settings=self.slab_generator_settings,
                               get_slab_settings=self.get_slab_settings,
                               bulk_dft_settings=self.bulk_dft_settings)

    def run(self, _testing=False):
        ''' Do not use `_testing=True` unless you are unit testing '''
        # Parse the possible adslab structures and find the one that matches
        # the site, shift, and top values we're looking for
        with open(self.input().path, 'rb') as file_handle:
            adslab_docs = pickle.load(file_handle)
        if self.adsorbate_name != '':
            doc = self._find_matching_adslab_doc(adslab_docs=adslab_docs,
                                                 adsorption_site=self.adsorption_site,
                                                 shift=self.shift,
                                                 top=self.top)
        # Hacky solution for finding empty slabs, where we don't care about the
        # site. This is only here because we are still (unfortunately) using
        # the adslab infrastructure to do slab calculations.
        else:
            doc = self._find_matching_adslab_doc_for_slab(adslab_docs=adslab_docs,
                                                          shift=self.shift,
                                                          top=self.top)
        atoms = make_atoms_from_doc(doc)

        # Create, package, and submit the FireWork
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'slab+adsorbate optimization',
                   'adsorbate': self.adsorbate_name,
                   'adsorbate_rotation': dict(self.rotation),
                   'adsorption_site': self.adsorption_site,
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'top': self.top,
                   'slab_repeat': doc['slab_repeat'],
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Let Luigi know that we've made the FireWork
        self._complete = True

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork

    @staticmethod
    def _find_matching_adslab_doc(adslab_docs, adsorption_site, shift, top):
        '''
        This helper function is used to parse through a list of documents
        created by the `GenerateAdslabs` task, and then find one that has
        matching values for site, shift, and top. If it doesn't find one, then
        it'll throw an error.  If there's more than one match, then it will
        just return the first one without any notification

        Args:
            adslab_docs     A list of dictionaryies created by
                            `GenerateAdslabs`
            adsorption_site A 3-long sequence of floats indicating the
                            Cartesian coordinates of the adsorption site
            shift           A float indicating the shift (i.e., slab
                            termination)
            top             A Boolean indicating whether or not the site is on
                            the top or the bottom of the slab
        Returns:
            doc     The first dictionary within the `adslab_docs` list that has
            matching site, shift, and top values
        '''
        for doc in adslab_docs:
            if np.allclose(doc['adsorption_site'], adsorption_site, atol=0.01):
                if math.isclose(doc['shift'], shift, abs_tol=0.01):
                    if doc['top'] == top:
                        return doc

        raise RuntimeError('You just tried to make an adslab FireWork rocket '
                           'that we could not enumerate. Try changing the '
                           'adsorption site, shift, top, or miller.')

    @staticmethod
    def _find_matching_adslab_doc_for_slab(adslab_docs, shift, top):
        '''
        This method is nearly identical to the `_find_matching_adslab_doc`
        method, except it ignores the adsorption site. This is used mainly in
        case you are trying to make a bare Adslab.

        Args:
            adslab_docs     A list of dictionaryies created by
                            `GenerateAdslabs`
            shift           A float indicating the shift (i.e., slab
                            termination)
            top             A Boolean indicating whether or not the site is on
                            the top or the bottom of the slab
        Returns:
            doc     The first dictionary within the `adslab_docs` list that has
            matching site, shift, and top values
        '''
        for doc in adslab_docs:
            if math.isclose(doc['shift'], shift, abs_tol=0.01):
                if doc['top'] == top:
                    return doc

        raise RuntimeError('You just tried to make an adslab FireWork rocket '
                           'that we could not enumerate. Try changing the '
                           'shift, top, or miller.')


class MakeSurfaceFW(FireworkMaker):
    '''
    This task will create and submit a surface calculation meant for surface
    energy calculations

    Args:
        atoms_doc       A dictionary created by feeding the surface to the
                        `gaspy.mongo.make_doc_from_atoms` function. This will
                        be used to rebuild the atoms object and then relax it.
        mpid            A string indicating the Materials Project ID of the
                        bulk you want to get a surface from
        miller_indices  A 3-tuple containing the three Miller indices of the
                        surface you want to find
        shift           A float indicating the shift of the surface---i.e., the
                        termination that pymatgen finds
        dft_settings    A dictionary containing your DFT settings for the
                        surface relaxation
    '''
    atoms_doc = luigi.DictParameter()
    mpid = luigi.Parameter()
    miller_indices = luigi.TupleParameter()
    shift = luigi.FloatParameter()
    dft_settings = luigi.DictParameter(SLAB_SETTINGS[DFT_CALCULATOR])

    def run(self, _testing=False):
        ''' Do not use `_testing=True` unless you are unit testing '''
        # Create, package, and submit the FireWork
        atoms = make_atoms_from_doc(unfreeze_dict(self.atoms_doc))
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'surface energy optimization',
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'num_slab_atoms': len(atoms),
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Let Luigi know that we've made the FireWork
        self._complete = True

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork


class MakeRismAdslabFW(MakeAdslabFW):
    '''
    This task will create and submit an adsorbate+slab (adslab) calculation
    using RISM. This is different from `MakeAdslabFW` because it uses the
    atomic positions of a given, potentially relaxed structure as a seed for
    the relaxation, which helps RISM converge more easily.

    Note that we still use a bunch of arguments like `adsorbate_name` or
    `adsorption_site`. These are used only to create a FireWork name that we
    can query later. These are not actually used to set atomic positions or
    identities. Those are set implicitly in the `atoms_dict` argument.

    Since this is a child class of `MakeAdslabFW`, you can reference that
    parent class for additional information.

    Additional args:
        atoms_dict      A dictionary created by
                        `gaspy.mongo.make_doc_from_atoms`, but with the 'calc',
                        'ctime', and 'mtime' key/value pairs removed (mainly so
                        the values don't mess up Luigi when Luigi tries to hash
                        datetime objects and whatnot).
    '''
    atoms_dict = luigi.DictParameter()

    def run(self, _testing=False):
        ''' Do not use `_testing=True` unless you are unit testing '''
        atoms = make_atoms_from_doc(self.atoms_dict)

        # Create, package, and submit the FireWork
        dft_settings = unfreeze_dict(self.dft_settings)
        fw_name = {'calculation_type': 'slab+adsorbate optimization',
                   'adsorbate': self.adsorbate_name,
                   'adsorbate_rotation': dict(self.rotation),
                   'adsorption_site': self.adsorption_site,
                   'mpid': self.mpid,
                   'miller': self.miller_indices,
                   'shift': self.shift,
                   'top': self.top,
                   'dft_settings': dft_settings}
        fwork = make_firework(atoms=atoms,
                              fw_name=fw_name,
                              dft_settings=dft_settings)
        _ = submit_fwork(fwork=fwork, _testing=_testing)    # noqa: F841

        # Let Luigi know that we've made the FireWork
        self._complete = True

        # Pass out the firework for testing, if necessary
        if _testing is True:
            return fwork
