GASpy is effectively a wrapper for FireWorks, Luigi, and MongoDB.
This means that the more adept you are with these tools, the more adept you will be at using GASpy.
This short guide will help you get started.


# Quick Start

## Enumerate catalog
Adsorption energy calculations are currently configured to only run on adsorption sites that you have already enumerated.
To enumerate a site, you first need to relax the bulk structure and then add it to the `catalog` Mongo collection.
An example script for doing this is included [here](../examples/populate_catalog.py).
Note that you will need to be inside a GASpy container for it to work.
That script will submit any bulk relaxation calculations you may need.
You will then need to wait for them to finish and then be [updated to the `atoms` collection.](../examples/update_collections.py).
Once the bulk relaxations are added to this database, you can rerun the [population script](../examples/populate_catalog.py) and it should add the sites to the catalog.

## Submit adsorption calculation
Afterwards, `gaspy.gasdb.get_catalog_docs` will return a list of dictionaries, where each dictionary represents one of the the enumerated sites.
Here is an example of one:

    {'adsorption_site': [3.6382062367640263e-16,
                         4.180407573039446e-16,
                         31.432295816723716],
     'coordination': 'Cu',
     'miller': [1, 1, 1],
     'mongo_id': ObjectId('5d378707e3a7f078783c4552'),
     'mpid': 'mp-30',
     'natoms': 4,
     'neighborcoord': ['Cu:Cu-Cu-Cu'],
     'shift': 0.16666666666666666,
     'top': True}

Here is an example of how you parse through these sites to submit a calculation:

    from gaspy.gasdb import get_catalog_docs
    from gaspy.tasks.metadata_calculators import submit_adsorption_calculations
    
    
    # Get all of the sites that we have enumerated
    docs = get_catalog_docs()
    
    # Pick the sites that we want to run. In this case, it'll be sites on
    # palladium (as per Materials Project ID 2, mp-2) on (111) facets.
    site_documents_to_calc = [doc for doc in all_site_documents
                              if (doc['mpid'] == 'mp-2' and
                                  doc['miller'] == [1, 1, 1])]
    
    adsorbate = 'CO'
    submit_adsorption_calculations((adsorbate='CO', catalog_docs=site_documents_to_calc))

## Reading data
As you may have noticed, we use the term "doc" in the GASpy API.
We [inherited this term from MongoDB](https://docs.mongodb.com/manual/core/document/).
This means that if you see a function that says `get_*_doc`, then it is reading information from the MongoDB you set up.
The primary functions of interest are all in the [`gaspy.gasdb`](../gaspy/gasdb.py) submodule, and include `get_catalog_docs`, `get_adsorption_docs`, `get_surface_docs`.
They get the enumerated adsorption sites; get the information about adsorption energies you've calculated; and get the information about surface energies you've calculated, respectively.
For example:  This is how you get a list of all of the `CO` adsorption energies you have:

    from gaspy.gasdb import get_adsorption_docs

    docs = get_adsorption_docs(adsorbate='CO')

Here is what one of these adsorption documents/dictionaries look like:

    {'adsorbate': 'CO',
     'coordination': 'Ni-Ni-Ni-Ni-Pt',
     'energy': -0.67283776999998416,
     'miller': [1, 0, 0],
     'mongo_id': ObjectId('5d38288f38bc9b2cc322e8ec'),
     'mpid': 'mp-945',
     'neighborcoord': ['Ni:Ni-Ni-Ni-Ni-Pt-Pt-Pt-Pt',
                       'Ni:Ni-Ni-Ni-Ni-Pt-Pt-Pt-Pt',
                       'Pt:Ni-Ni-Ni-Ni-Ni-Ni-Ni-Ni-Pt-Pt-Pt',
                       'Ni:Ni-Ni-Ni-Ni-Pt-Pt-Pt-Pt',
                       'Ni:Ni-Ni-Ni-Ni-Pt-Pt-Pt-Pt'],
     'shift': 0.,
     'top': False}

## Getting [`Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects
If you are accustomed to working with `ase.Atoms` objects, then you can convert any document from the GASpy collections into an `Atoms` object one using the [`gaspy.mongo.make_atoms_from_doc`](../gaspy/mongo.py) function.
For example:

    from gaspy.mongo import make_atoms_from_doc
    from gsapy.gasdb import get_adsorption_docs


    docs = get_adsorption_docs(extra_projections={'atoms': '$atoms', 'results': '$results'})
    atoms = make_atoms_from_doc(docs[0])


# Advanced usage

We cannot stress this enough:  **the better you at using Luigi, MongoDB, and FireWorks, the better you will be at using GASpy**.

## Luigi

The primary tasks of interest are mostly inside the [`gaspy.tasks.metadata_calculators`](../gaspy/tasks/metadata_calculators.py) submodule.
They currently include `CalculateAdsorptionEnergy` and `CalculateSurfaceEnergy`.
In both Luigi and GASpy, you must [schedule these tasks](https://luigi.readthedocs.io/en/stable/running_luigi.html#running-from-python-code).
GASpy has a light wrapper for doing this scheduling:  [`schedule_tasks`](../gaspy/tasks/core.py).

### Finding failed dependencies
If a task fails, then you must read the traceback to see why it failed.
The traceback is likely to show you exactly why something went wrong.
If the traceback does not show you the error in an apparent way, then we recommend finding the first task in the dependency tree that failed.
We recommend using [Luigi's API](https://luigi.readthedocs.io/en/stable/central_scheduler.html) for finding the first failed task.
You will need to [tunnel](https://www.ssh.com/ssh/tunneling/) to your Luigi Daemon to use this API---e.g., `ssh -gnNT -L 8082:host.ip.address:8082 username@host.address`.

Or you could stay inside of Python.
If you execute the `task.requires()` method of a task, it will give you its upstream dependencies.
Then if you execute the `task.complete()` method of a task, it will tell you whether the task is complete.
You can keep tracing the tasks backwards until you find the first failure, and then investigate.

### Dynamic dependencies
GASpy makes use of the [dynamic dependencies](https://luigi.readthedocs.io/en/stable/tasks.html#dynamic-dependencies) feature of Luigi.
For example:  When we ask GASpy to [find a calculation](../gaspy/tasks/calculation_finders.py), GASpy will first look in the Mongo database.
If the calculation is there, then the "finder" task will simply return as `complete`.
But if the calculation is not there, then the "finder" may require a [make_fireworks](../gaspy/tasks/make_fireworks.py) dependency.
As per the Luigi API, dynamic dependencies are returned from the `task.run()` method, not the `task.requires()` method. This is useful to know when debugging and/or trying to find upstream dependencies.

As a side-note:  The "finder" tasks will return as `task.complete() == True` even if a calculation was not found.
This is an artifact of how we use both static and dynamic dependencies, and we have not yet found a way to avoid it.
Please be wary of this when debugging.

### Debugging tasks
Since all of the work is delegated to the daemon, you will not have direct access to stack for debugging.
This can be very annoying for those of us accustomed to using [pdb](https://docs.python.org/3/library/pdb.html).
To work around this, we created the [`gaspy.tasks.core.run_task`](../gaspy/tasks/core.py) function.
This function works similar to `schedule_tasks` function, except it takes only one task at a time and executes it locally in a rather hacky fashion.
Since `run_task` executes the task locally, you'll have access to the stack and using `pdb` will be successful.
We ofter combine `run_task` with the [`%debug`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug) [Jupyter](https://jupyter.org/) command.

Be warned:  The `run_task` function is not as stable as the `schedule_tasks` function, and is therefore not guaranteed to work as intended in all situations.
Please use it only for debugging purposes.

### Debugging relaxations
If you realize that one of you tasks is not working, then it might be because the DFT calculation is failing.
If you realize this and want to figure out why it's failing, you'll need to go to the directory where it ran and parse the error logs.
To find this directory, simply `schedule_tasks([task])` the job you want to investigate.
During the process of trying to run the task, GASpy will find the fizzled job and then report the FireWorks ID (fwid), the host that the job ran on, and its launch directory.

Note that if you schedule a task and one of the jobs [fizzled](https://materialsproject.github.io/fireworks/failures_tutorial.html#error-during-run-a-fizzled-firework)/failed, then GASpy may try to redo the calculation.
This is by design, because in an automated workflow sometimes things fail for random reasons---e.g., server crashed, and we want to automatically resubmit those calculations.
If you do not want to re-try a calculation, then give the `gaspy.tasks.metadata_calculators.*` task the `max_fizzles=1` argument.
If you do this, then GASpy will refrain from rerunning jobs.


## Mongo

### Collection handling
The primary `get_catalog_docs`, `get_adsorption_docs`, and `get_surface_docs` functions use [aggregation](https://docs.mongodb.com/manual/aggregation/) to parse through the database.
If you are adept with Mongo and want to pull the information yourself, then you can simply pull all the information however you want.
We have also provided a small helper function in [`gaspy.gasdb.get_mongo_collection`](../gaspy/gasdb.py).
This function will give you child class of the pymongo [`collection`](https://api.mongodb.com/python/current/api/pymongo/collection.html) object that is already authenticated with your GASpy Mongo database credentials and has an `__exit__` method.
Thus proper usage would be like this:
    from gaspy.gasdb import get_mongo_collection

    with get_mongo_collection('atoms') as collection:
        cursor = collection.find({}, {'fwname': 1, '_id': 0})
        docs = list(cursor)
This will get you all of the unparsed run information of all of your successful DFT calculations.

### Document structures
If you want to start using Mongo directly, it's helpful to know what the structure of the document will look like first.
You can either look it up directly, or you can reference the sample `json` files we have in this directory.

We should note that the documents in the `atoms` collection have the `atoms` field and the `initial_configuration` field.
If you apply the [`gaspy.mongo.make_atoms_from_doc`](../gaspy/mongo.py) function to an `atoms` document raw, then it will give you the relaxed structure.
But if you apply the `make_atoms_from_doc` function to the `document['initial_configuration']`, then it will give you the unrelaxed structure.
The same applies to documents from the `adsorption` collection.

If you want the structures from the `surface_energy` collection, then you need to look inside the `surface_energy_document['surface_structures']` field.
This field/sequence will have three dictionaries inside of it.
The first dictionary contains both the final and initial structure of the thinnest slab in the surface energy calculation, where you can get the `ase.Atoms` structures just like you do with the `atoms` or `adsorption` documents.
The second dictionary is the same as the first dictionary, but for the second-thinnest slab.
The third and final dictionary corresponds to the thickest slab.


## FireWorks

If you use GASpy and FireWorks naively and within a team, then the default behaviour may involve everyone on the team pooling both their data and their computing time.
If you do not want to pool your computing time, then you should each have your own FireWorks launchers.
When you create a launcher, you need to specify a `fworker.yaml` file.
Inside your `fworker.yaml` files, you can have a `query` argument.
This is where you can add a MongoDB [query](https://docs.mongodb.com/manual/tutorial/query-documents/) on your FireWorks database.
This means that your launcher will only run jobs that match the query you put in there.
Therefore, if you want to run only jobs that you have submitted, then simply enter `query: '{"name.user": "my_name"}'` into the `fworker.yaml`.
More advanced things can be done with FireWorks, but this is the main thing to consider when using GASpy on a team.
