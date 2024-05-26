import os 
import glob
import veloxchem as vlx


def find_itp_file(path,grofile_name):
    itp_files = glob.glob(os.path.join(path, '**/'+grofile_name+'*.itp'), recursive=True)
    return os.path.basename(itp_files[0]) if itp_files else None

def find_linkerxyz_file(path):
    
    xyz_files = glob.glob(os.path.join(path, '*.xyz'), recursive=True)
    return os.path.abspath(xyz_files[0]) if xyz_files else None
    #return os.path.basename(xyz_files[0]) if xyz_files else None


def mvff_files2subfolder(grofile_name):
    subfolder = os.path.abspath ( '')+'/itps/' 
    os.makedirs(subfolder,exist_ok=True)
    ff_files = glob.glob(os.path.join(grofile_name+'*'), recursive=True)
    for f in ff_files:
        os.rename(f,subfolder+f)

def ff_gen(path,charge,basis,grofile_name,residue_name):
    #maefilestrip = mae.strip(".01.mae")
    #newpath = os.path.abspath ( '')+'/'+str(maefilestrip)+'/' 
    xyzfile = find_linkerxyz_file(path)
#Import molecule and set molecule properties
    molecule = vlx.Molecule.read_xyz_file(xyzfile)
    molecule.set_charge(charge)  #TODO:
    #molecule.set_multiplicity(2) #TODO:
    basis = vlx.MolecularBasis.read(molecule, basis) 
#Set scf_driver
    scf_drv = vlx.ScfUnrestrictedDriver() #TODO:
    #scf_drv.guess_unpaired_electrons = '60(1)' #TODO:
    scf_drv.conv_thresh = 1e-2
    scf_drv.max_iter = 500
    scf_drv.xcfun = 'b3lyp'
    scf_results = scf_drv.compute(molecule, basis)
##Generate the force field with the scf results
    ff_gen = vlx.ForceFieldGenerator()
    ff_gen.ostream.mute()
    ff_gen.create_topology(molecule, basis, scf_results)
    #ff_gen.create_topology(molecule, no_resp=True)
    ff_gen.write_gromacs_files(grofile_name,residue_name)
    #molecule.show(atom_indices=True)
    
    #itp_files = glob.glob(os.path.join(path, '**/*.itp'), recursive=True)

    #os.rename($grofile_name*, destination_path) $grofile_name* $newpath
    mvff_files2subfolder(grofile_name)