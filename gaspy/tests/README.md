# `gaspy.tests` Content

Welcome to the gaspy testing submodule. The structure of this folder mostly
mimics the structure of the gaspy folders, so you should be able to find the
appropriate tests at the analogous locations in this submodule. There are
some differences that warrant an explanation, though.

## `gaspy.tests.learning_tests`

A learning test is a type of unit test that is devoted to testing the
functionality of something that the code base depends on. For example:  We
use pymatgen, but pymatgen is an actively changing repository. Thus its
functionality may change and then affect how GASpy operates. Learning tests
double-check these changes. So if a GASpy developer decides that some
dependency is important enough to warrant its own test control, then they
are free to add a learning test to this submodule.

## `gaspy.tests.test_cases`

GASpy operates on atoms objects a lot. But not all atoms are built the same.
This submodule is virtually a cache of various atoms objects that can be used
during testing, along with a small API to fetch the objects. For example, you
can get a Cu FCC structure via:

```
from gaspy.tests import test_cases
atoms = test_cases.get_bulk_atoms('Cu_FCC.traj')
```

The types of atoms you can get are listed within the subfolders of
`gaspy.tests.test_cases.*`. You can add more yourself if you want to create
more test cases, too.

## `gaspy.tests.set_up_mongo`

Your Mongo database is probably going to explode if you populate it with enough
data. It'll be so big that doing unit tests on the database will get very
slow, and slow unit tests = not performed unit tests. So we set up small
Mongo collections and do unit testing on them, instead. The functions in this
submodule do this for you. This is how you can create and set up some
unit testing Mongo collections from Python:

```
from gaspy.tests.set_up_mongo import create_and_populate_unit_testing_collection

for collection_tag in ['unit_testing_adsorption', 'unit_testing_catalog', 'unit_testing_surface_energy']:
    create_and_populate_unit_testing_collection(collection_tag)
```

Note that these require a Mongo server set up and your `gaspyrc.json` to be
populated accordingly.

## `gaspy.tests.regression_baselines`

In computational science, it is common for a function to not have a "correct
output". Thus the main way to control it with tests is to see if the output
of a function changes. This is called a regression test.

This folder is a cache for all of GASpy's regression test results. The
structure of this folder should mimic the structure of the `gaspy.tests` folder
itself so that you know where to find the appropriate caches for a test.


# Usage

We use [pytest](https://docs.pytest.org/en/latest/). To do a full test on
GASpy, you need to have Mongo set up (see above). Then you can do this from the
command line:

```
cd GASpy/gaspy/tests
pytest
```

You can also test a specific submodule:

```
cd GASpy/gaspy/tests
pytest utils_test.py
```

Remember that you can use [pdb](https://docs.python.org/3/library/pdb.html) by
supplying the `--pdb` argument when calling `pytest`. We recommend that you only run
pytest from within the `gaspy/tests` folder or below to ensure that pytest uses
the `pytest.ini` files.

## Regression testing

As per the `pytest.ini` files, we ignore unit tests that are
[marked](https://docs.pytest.org/en/latest/example/markers.html) with the
custom mark `baseline`. "Unit tests" that are marked like this are not actually
unit tests. They are functions that update the regression testing baselines for
various unit testing functions. By GASpy convention, these functions are
prefixed with `test_to_create_*`.

If you change a function and acknowledge that the regression baseline should change,
then find the appropriate baseline function and execute it. You can also run it via
pytest like this:

```
pytest gaspy/module.py::function_marked_with_baseline -m baseline
```

...or if you want to redo the baselines for an entire submodule:

```
pytest gaspy/module.py -m baseline
```

...or if you just want to redo all of the baselines:

```
pytest gaspy -m baseline
```

If you plan to build regression tests, then please refer to the
`gaspy.tests.regression_baselines` section above and use the current regression
tests as templates.
