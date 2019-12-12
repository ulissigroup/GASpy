'''
This submodule is part of GASpy. It is meant to be used by FireWorks to perform
VASP calculations.
'''

__authors__ = ['Zachary W. Ulissi', 'Kevin Tran']
__emails__ = ['zulissi@andrew.cmu.edu', 'ktran@andrew.cmu.edu']

import os
import uuid
import numpy as np
import ase.io
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import BFGS
from ase.calculators.vasp import Vasp2
from ase.calculators.singlepoint import SinglePointCalculator as SPC

# TODO:  Need to handle setting the pseudopotential directory, probably in the
# submission config if it stays constant? (vasp_qadapter.yaml)


def runVasp(fname_in, fname_out, vasp_flags):
    '''
    This function is meant to be sent to each cluster and then used to run our
    rockets. As such, it has algorithms to run differently depending on the
    cluster that is trying to use this function.

    Args:
        fname_in    A string indicating the file name of the initial structure.
                    This file should be readable by `ase.io.read`.
        fname_out   A string indicating the name of the file you want to save
                    the final, relaxed structure to.
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        atoms_str   A string-formatted name for the atoms
        traj_hex    A string-formatted hex enocding of the entire relaxation
                    trajectory
        energy      A float indicating the potential energy of the final image
                    in the relaxation [eV]
    '''
    # Read the input atoms object
    atoms = ase.io.read(str(fname_in))

    # Perform the relaxation
    final_image = _perform_relaxation(atoms, vasp_flags, fname_out)

    # Parse and return output
    atoms_str = str(atoms)
    traj_hex = open('all.traj', 'r').read().encode('hex')
    energy = final_image.get_potential_energy()
    return atoms_str, traj_hex, energy


def _perform_relaxation(atoms, vasp_flags, fname_out):
    '''
    This function will perform the DFT relaxation while also saving the final
    image for you.

    Args:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
        fname_out   A string indicating the file name you want to use when
                    saving the final, relaxed structure
    Returns:
        atoms   The relaxed `ase.Atoms` structure
    '''
    # Initialize some things before we run
    atoms, vasp_flags = _clean_up_vasp_inputs(atoms, vasp_flags)
    vasp_flags = _set_vasp_command(vasp_flags)

    # Detect whether or not there are constraints that cannot be handled by VASP
    allowable_constraints = {'FixAtoms'}
    vasp_compatible = True
    for constraint in atoms.constraints:
        if constraint.todict()['name'] not in allowable_constraints:
            vasp_compatible = False
            break

    # Run with VASP by default
    if vasp_compatible:
        final_image = _relax_with_vasp(atoms, vasp_flags)
    # If VASP can't handle it, then use ASE/VASP together
    else:
        final_image = _relax_with_ase(atoms, vasp_flags)

    # Delete some files to save disk space
    _delete_electronic_log_files()

    # Save the last image
    final_image.write(str(fname_out))
    return final_image


def _clean_up_vasp_inputs(atoms, vasp_flags):
    '''
    There are some VASP settings that are used across all our clusters. This
    function takes care of these settings.

    Arg:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        atoms       `ase.Atoms` object of the structure we want to relax, but
                    with the unit vectors fixed (if needed)
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    # Check that the unit vectors obey the right-hand rule, (X x Y points in
    # Z). If not, then flip the order of X and Y to enforce this so that VASP
    # is happy.
    if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0:
        atoms.set_cell(atoms.cell[[1, 0, 2], :])

    # Set the pseudopotential type by setting 'xc' in Vasp()
    if vasp_flags['pp'].lower() == 'lda':
        vasp_flags['xc'] = 'lda'
    elif vasp_flags['pp'].lower() == 'pbe':
        vasp_flags['xc'] = 'PBE'

    # Push the pseudopotentials into the OS environment for VASP to pull from
    pseudopotential = vasp_flags['pp_version']
    os.environ['VASP_PP_PATH'] = os.environ['VASP_PP_BASE'] + '/' + str(pseudopotential) + '/'
    del vasp_flags['pp_version']

    return atoms, vasp_flags


def _set_vasp_command(vasp_flags):
    '''
    This function assigns the appropriate call to VASP to the `$VASP_COMMAND`
    variable. The command depends on what cluster we are running on. This
    function figures that out automatically.

    Arg:
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    # Figure out where we're running based on environment variables
    if 'SLURM_CLUSTER_NAME' in os.environ:
        cluster_name = os.environ['SLURM_CLUSTER_NAME']
    elif 'PBS_O_HOST' in os.environ:
        cluster_name = os.environ['PBS_O_HOST'].split()[0]
    else:
        raise RuntimeError('Could not figure out what machine this job is '
                           'running on and therefore could not figure out '
                           'how to run VASP correctly. If you would like to '
                           'add a subroutine for your cluster, please send a '
                           'pull request to '
                           'https://github.com/ulissigroup/GASpy')

    # Define which functions to use for which clusters
    command_makers = {'cori': __make_cori_vasp_command,
                      'arjuna': __make_arjuna_vasp_command,
                      'gilgamesh': __make_gilgamesh_vasp_command}

    # Make the appropriate command to call VASP and then assign it to the
    # environment so that VASP can pick it up automatically.
    vasp_command, vasp_flags = command_makers[cluster_name](vasp_flags)
    os.environ['VASP_COMMAND'] = vasp_command
    return vasp_flags


def __make_cori_vasp_command(vasp_flags):
    '''
    Makes a VASP command to use on Cori

    Arg:
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        command     The command that should be executed in bash to run VASP
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    vasp_executable = 'vasp_std'

    # Figure out the number of processes
    try:
        n_processors = int(os.environ['SLURM_NPROCS'])
    except KeyError:
        n_processors = 1

    # If we're on a Haswell node...
    if os.environ['CRAY_CPU_TARGET'] == 'haswell' and 'knl' not in os.environ['PATH']:
        n_nodes = int(os.environ['SLURM_NNODES'])
        vasp_flags['kpar'] = n_nodes
        command = 'srun -n %d %s' % (n_processors, vasp_executable)

    # If we're on a KNL node...
    elif 'knl' in os.environ['PATH']:
        command = 'srun -n %d -c8 --cpu_bind=cores %s' % (n_processors*32, vasp_executable)
        vasp_flags['ncore'] = 1

    return command, vasp_flags


def __make_arjuna_vasp_command(vasp_flags):
    '''
    Makes a VASP command to use on Arjuna

    Arg:
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        command     The command that should be executed in bash to run VASP
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    # Figure out the number of processes
    try:
        n_processors = int(os.environ['SLURM_NPROCS'])
    except KeyError:
        n_processors = 1

    # If this is a GPU job...
    if os.environ['CUDA_VISIBLE_DEVICES'] != 'NoDevFiles':
        vasp_flags['ncore'] = 1
        vasp_flags['kpar'] = 16
        vasp_flags['nsim'] = 8
        vasp_flags['lreal'] = 'Auto'
        vasp_executable = 'vasp_gpu'
        command = 'mpirun -np %i %s' % (n_processors, vasp_executable)

    # If this is a CPU job...
    else:
        vasp_executable = 'vasp_std'
        if n_processors > 16:
            vasp_flags['ncore'] = 4
            vasp_flags['kpar'] = 4
        else:
            vasp_flags['kpar'] = 1
            vasp_flags['ncore'] = 4
        command = 'mpirun -np %i %s' % (n_processors, vasp_executable)

    return command, vasp_flags


def __make_gilgamesh_vasp_command(vasp_flags):
    '''
    Makes a VASP command to use on Gilgamesh

    Arg:
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        command     The command that should be executed in bash to run VASP
        vasp_flags  A modified version of the 'vasp_flags' argument
    '''
    os.environ['PBS_SERVER'] = 'gilgamesh.cheme.cmu.edu'
    vasp_flags['npar'] = 4
    vasp_executable = '/home-research/zhongnanxu/opt/vasp-5.3.5/bin/vasp-vtst-beef-parallel'
    n_processors = len(open(os.environ['PBS_NODEFILE']).readlines())
    command = 'mpirun -np %i %s' % (n_processors, vasp_executable)
    return command, vasp_flags


def _relax_with_ase(atoms, vasp_flags):
    '''
    Instead of letting VASP handle the relaxation autonomously, we instead use
    VASP only as an eletronic structure calculator and use ASE's BFGS to
    perform the atomic position optimization.

    Note that this will also write the trajectory to the 'all.traj' file and
    save the log file as 'relax.log'.

    Args:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        atoms   The relaxed `ase.Atoms` structure
    '''
    vasp_flags['ibrion'] = 2
    vasp_flags['nsw'] = 0
    calc = Vasp2(**vasp_flags)
    atoms.set_calculator(calc)
    optimizer = BFGS(atoms, logfile='relax.log', trajectory='all.traj')
    optimizer.run(fmax=vasp_flags['ediffg'] if 'ediffg' in vasp_flags else 0.05)
    return atoms


def _relax_with_vasp(atoms, vasp_flags):
    '''
    Perform a DFT relaxation with VASP and then write the trajectory to the
    'all.traj' file and save the log file.

    Args:
        atoms       `ase.Atoms` object of the structure we want to relax
        vasp_flags  A dictionary of settings we want to pass to the `Vasp2`
                    calculator
    Returns:
        atoms   The relaxed `ase.Atoms` structure
    '''
    # Run the calculation
    calc = Vasp2(**vasp_flags)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    # Read the trajectory from the output file
    images = []
    for atoms in ase.io.read('vasprun.xml', ':'):
        image = atoms.copy()
        image = image[calc.resort]
        image.set_calculator(SPC(image,
                                 energy=atoms.get_potential_energy(),
                                 forces=atoms.get_forces()[calc.resort]))
        images += [image]

    # Write the trajectory
    with TrajectoryWriter('all.traj', 'a') as tj:
        for atoms in images:
            tj.write(atoms)
    return images[-1]


def _delete_electronic_log_files():
    '''
    VASP's electronic log files can get big. This function will delete them for
    you with the intent of saving hard drive space.
    '''
    for file_ in ['CHGCAR', 'WAVECAR', 'CHG']:
        try:
            os.remove(file_)
        except OSError:
            pass


def atoms_to_hex(atoms):
    '''
    Turn an atoms object into a hex string so that we can pass it through fireworks

    Arg:
        atoms   The `ase.Atoms` object that you want to hex encode
    Returns:
        _hex    A hex string of the `ase.Atoms` object
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


def hex_to_file(file_name, hex_):
    '''
    Write a hex string into a file. One application is to unpack hexed atoms
    pobjects in local fireworks job directories

    Args:
        file_name   A string indicating the name of the file you want to write
                    to
        hex_        A hex string of the object you want to write to the file
    '''
    # Dump the hex encoded string to a local file
    with open(file_name, 'w') as fhandle:
        fhandle.write(hex_.decode('hex'))
