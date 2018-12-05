import os
import uuid
import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import BFGS
from ase.calculators.vasp import Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC
# noqa: E731

# Need to handle setting the pseudopotential directory, probably in the submission
# config if it stays constant? (vasp_qadapter.yaml)


def runVasp(fname_in, fname_out, vaspflags, npar=4):
    '''
    This function is meant to be sent to each cluster and then used to run our rockets.
    As such, it has algorithms to run differently depending on the cluster that is trying
    to use this function.

    Inputs:
        fname_in
        fname_out
        vaspflags
        npar
    '''
    fname_in = str(fname_in)
    fname_out = str(fname_out)

    # read the input atoms object
    atoms = read(str(fname_in))

    # Check that the unit vectors obey the right-hand rule, (X x Y points in Z) and if not
    # Flip the order of X and Y to enforce this so that VASP is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    vasp_cmd = 'vasp_std'
    os.environ['PBS_SERVER'] = 'gilgamesh.cheme.cmu.edu'

    if 'PBS_NODEFILE' in os.environ:
        NPROCS = NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
    elif 'SLURM_CLUSTER_NAME' in os.environ:
        if 'SLURM_NPROCS' in os.environ:
            # We're on cori haswell
            NPROCS = int(os.environ['SLURM_NPROCS'])
        else:
            # we're on cori KNL, just one processor
            NPROCS = 1

    # If we're on Gilgamesh...
    if 'PBS_NODEFILE' in os.environ and os.environ['PBS_SERVER'] == 'gilgamesh.cheme.cmu.edu':
        vaspflags['npar'] = 4
        vasp_cmd = '/home-research/zhongnanxu/opt/vasp-5.3.5/bin/vasp-vtst-beef-parallel'
        NPROCS = NPROCS = len(open(os.environ['PBS_NODEFILE']).readlines())
        mpicall = lambda x, y: 'mpirun -np %i %s' % (x, y)  # noqa: E731

    # If we're on Arjuna...
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME'] == 'arjuna':
        # If this is a GPU job...
        if os.environ['CUDA_VISIBLE_DEVICES'] != 'NoDevFiles':
            vaspflags['ncore'] = 1
            vaspflags['kpar'] = 16
            vaspflags['nsim'] = 8
            vaspflags['lreal'] = 'Auto'
            vasp_cmd = 'vasp_gpu'
            mpicall = lambda x, y: 'mpirun -np %i %s' % (x, y)  # noqa: E731
        # If this is a CPU job...
        else:
            if NPROCS > 16:
                vaspflags['ncore'] = 4
                vaspflags['kpar'] = 4
            else:
                vaspflags['kpar'] = 1
                vaspflags['ncore'] = 4
            mpicall = lambda x, y: 'mpirun -np %i %s' % (x, y)  # noqa: E731

    # If we're on Cori, use SLURM. Note that we decrease the priority by 1000
    # in order to prioritize other things higher, such as modeling and prediction
    # in GASpy_regression
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME'] == 'cori':
        # If we're on a Haswell node...
        if os.environ['CRAY_CPU_TARGET'] == 'haswell' and 'knl' not in os.environ['PATH']:
            NNODES = int(os.environ['SLURM_NNODES'])
            vaspflags['kpar'] = NNODES
            mpicall = lambda x, y: 'srun -n %d %s' % (x, y)  # noqa: E731
        # If we're on a KNL node...
        elif 'knl' in os.environ['PATH']:
            mpicall = lambda x, y: 'srun -n %d -c8 --cpu_bind=cores %s' % (x*32, y)  # noqa: E731
            vaspflags['ncore'] = 1

    # Set the pseudopotential type by setting 'xc' in Vasp()
    if vaspflags['pp'].lower() == 'lda':
        vaspflags['xc'] = 'lda'
    elif vaspflags['pp'].lower() == 'pbe':
        vaspflags['xc'] = 'PBE'

    pseudopotential = vaspflags['pp_version']
    os.environ['VASP_PP_PATH'] = os.environ['VASP_PP_BASE'] + '/' + str(pseudopotential) + '/'
    del vaspflags['pp_version']

    os.environ['VASP_COMMAND'] = mpicall(NPROCS, vasp_cmd)

    # Detect whether or not there are constraints that cannot be handled by VASP
    allowable_constraints = ['FixAtoms']
    constraint_not_allowable = [constraint.todict()['name']
                                not in allowable_constraints
                                for constraint in atoms.constraints]
    vasp_incompatible_constraints = np.any(constraint_not_allowable)

    # If there are incompatible constraints, we need to switch to an ASE-based optimizer
    if vasp_incompatible_constraints:
        vaspflags['ibrion'] = 2
        vaspflags['nsw'] = 0
        calc = Vasp2(**vaspflags)
        atoms.set_calculator(calc)
        qn = BFGS(atoms, logfile='relax.log', trajectory='all.traj')
        qn.run(fmax=vaspflags['ediffg'] if 'ediffg' in vaspflags else 0.05)
        finalimage = atoms

    else:
        # set up the calculation and run
        calc = Vasp2(**vaspflags)
        atoms.set_calculator(calc)

        # Trigger the calculation
        atoms.get_potential_energy()

        atomslist = []
        for atoms in read('vasprun.xml', ':'):
            catoms = atoms.copy()
            catoms = catoms[calc.resort]
            catoms.set_calculator(SPC(catoms,
                                      energy=atoms.get_potential_energy(),
                                      forces=atoms.get_forces()[calc.resort]))
            atomslist += [catoms]

        # Get the final trajectory
        finalimage = atoms

        # Write a traj file for the optimization
        tj = TrajectoryWriter('all.traj', 'a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close()

    # Write the final structure
    finalimage.write(fname_out)

    # Write a text file with the energy
    with open('energy.out', 'w') as fhandle:
        fhandle.write(str(finalimage.get_potential_energy()))

    try:
        os.remove('CHGCAR')
    except OSError:
        pass

    try:
        os.remove('WAVECAR')
    except OSError:
        pass

    try:
        os.remove('CHG')
    except OSError:
        pass

    return str(atoms), open('all.traj', 'r').read().encode('hex'), finalimage.get_potential_energy()


def atoms_to_hex(atoms):
    '''
    Turn an atoms object into a hex string so that we can pass it through fireworks

    Input:
        atoms   The ase.Atoms object that you want to hex encode
    '''
    # We need to write the atoms object into a file before encoding it. But we don't
    # want multiple calls to this function to interfere with each other, so we generate
    # a random file name via uuid to reduce this risk. Then we delete it.
    fname = str(uuid.uuid4()) + '.traj'
    atoms.write(fname)
    with open(fname) as fhandle:
        _hex = fhandle.read().encode('hex')
        os.remove(fname)
    return _hex


def hex_to_file(fname_out, atomHex):
    '''
    Write a hex string into a file. One application is to unpack hexed atoms objects in
    local fireworks job directories
    '''
    # Dump the hex encoded string to a local file
    with open(fname_out, 'w') as fhandle:
        fhandle.write(atomHex.decode('hex'))
