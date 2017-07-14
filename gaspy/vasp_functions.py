import os
import uuid
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter
from vasp import Vasp
from ase.optimize import QuasiNewton, BFGS
from vasp.vasprc import VASPRC
import os
import numpy as np

#Need to handle setting the pseudopotential directory, probably in the submission config if it stays constant? (vasp_qadapter.yaml)

def runVasp(fname_in,fname_out,vaspflags,npar=4):
    fname_in=str(fname_in)
    fname_out=str(fname_out)

    #read the input atoms object
    atoms=read(str(fname_in))
    
    #update vasprc file to set mode to "run" to ensure that this runs immediately
    Vasp.vasprc(mode='run')

    #set ppn>1 so that it knows to do an mpi job, the actual ppn will guessed by Vasp module
    Vasp.VASPRC['queue.ppn']=2
    
    vasp_cmd='vasp_std'

    if 'PBS_SERVER' in os.environ and os.environ['PBS_SERVER']=='gilgamesh.cheme.cmu.edu':
        #We're on gilgamesh
        vaspflags['NPAR']=4
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME']=='arjuna':
        #We're on arjuna
        if os.environ['CUDA_VISIBLE_DEVICES']!='NoDevFiles':
            #We have a GPU job on arjuna
            vaspflags['NCORE']=1
            vaspflags['KPAR']=16
            vaspflags['NSIM']=8
            vaspflags['lreal']='Auto'
            vasp_cmd='vasp_gpu'
            VASPRC['system.mpicall']=lambda x,y: 'mpirun -np %i %s'%(x,y)
        else:
            #We're running CPU only
            vaspflags['NCORE']=4
            vaspflags['KPAR']=4
            print('we found arjuna cpu!')
            VASPRC['system.mpicall']=lambda x,y: 'mpirun -np %i %s'%(x,y)
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME']=='cori':
        #We're on cori
        if os.environ['CRAY_CPU_TARGET']=='haswell':
            #We're on a haswell CPU node
            NNODES=int(os.environ['SLURM_NNODES'])
            vaspflags['KPAR']=NNODES
            VASPRC['system.mpicall']=lambda x,y: 'srun -n %d %s'%(x,y)
        elif os.environ['CRAY_CPU_TARGET']=='knl':
            VASPRC['system.mpicall']=lambda x,y: 'srun -n %d -c8 --cpu_bind=cores %s'%(x*32,y)
            vaspflags['NCORE']=1

    #Set the pseudopotential type by setting 'xc' in Vasp() 
    if vaspflags['pp'].lower()=='lda':
        vaspflags['xc']='lda'
    elif vaspflags['pp'].lower()=='pbe':
        vaspflags['xc']='PBE'
 
    #if 'gga' not in vaspflags:
    #    vaspflags['xc']='lda'
    #elif vaspflags['gga'] in ['RP','BF','PE']:
    #    vaspflags['xc']='pbe'
        
    pseudopotential=vaspflags['pp_version']
    os.environ['VASP_PP_PATH']=os.environ['VASP_PP_BASE']+'/'+str(pseudopotential)+'/'
    VASPRC['vasp.executable.parallel']=vasp_cmd

    #Detect whether or not there are constraints that cannot be handled by VASP
    allowable_constraints=['FixAtoms']
    constraint_not_allowable = [constraint.todict()['name'] not in allowable_constraints for constraint in atoms.constraints]
    vasp_incompatible_constraints = np.any(constraint_not_allowable)

    #If there are incompatible constraints, we need to switch to an ASE-based optimizer
    if vasp_incompatible_constraints:
        vaspflags['ibrion']=2
        vaspflags['nsw']=0
        calc=Vasp('./',atoms=atoms,**vaspflags)
        atoms.set_calculator(calc)
        #calc.read_results()
        #print(atoms.get_forces())
        #print(atoms.get_potential_energy())
        qn=BFGS(atoms,logfile='relax.log',trajectory='all.traj')
        #qn=BFGS(atoms,logfile='relax.log',trajectory='all.traj',force_consistent=False)
        qn.run(fmax=vaspflags['ediffg'] if 'ediffg' in vaspflags else 0.05)
        finalimage=atoms

    else:

        #set up the calculation and run
        calc=Vasp('./',atoms=atoms,**vaspflags)
        calc.read_results()
        
        #Get the final trajectory
        atomslist=calc.traj
        finalimage=atomslist[-1]

        #Write a traj file for the optimization
        tj=TrajectoryWriter('all.traj','a')
        for atoms in atomslist:
            print('writing trajectory file!')
            print(atoms)
            tj.write(atoms)
        tj.close() 

    #Write the final structure
    finalimage.write(fname_out)
    
    #Write a text file with the energy
    with open('energy.out','w') as fhandle:
        fhandle.write(str(finalimage.get_potential_energy()))

    return str(atoms),open('all.traj','r').read().encode('hex'),finalimage.get_potential_energy()


def atoms_to_hex(atoms):
    '''
    Turn an atoms object into a hex string so that we can pass it through fireworks

    Input:
        atoms   The ase.Atoms object that you want to hex encode
    '''
    # We need to write the atoms object into a file before encoding it. But we don't
    # want multiple calls to this function to interfere with each other, so we generate
    # a random file name via uuid to reduce this risk. Then we delete it.
    with stri(uuid.uuid4()) + '.traj' as fname:
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
