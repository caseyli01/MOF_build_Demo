from MOF_build.functions.rotate import rotate
from MOF_build.functions.read import read
from MOF_build.functions.output import output
import numpy as np
df_cut_ori = read.pdb('MOF_build/lib/PCN222_lib_cut.lib')
array = df_cut_ori.loc[:,['x','y','z']].to_numpy()
any_tdx, any_tdy, any_tdz = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 2]))
dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
carte_basis = np.vstack([dx, dy, dz])
    # for uio66 fcb
tric_basis = rotate.coordinate_transfer(any_tdx, any_tdy, any_tdz, carte_basis)
O1=array[36-1,:]
O2=array[1-1,:]
O3=array[43-1,:]
O4=array[22-1,:]
O5=array[15-1,:]
top=0.5*(O1+O2)
bottom=0.5*(O3+O4)
center = 0.5*(top+bottom)
v2_file=top-bottom
v1_file = O5-O1
v1_frame=tric_basis[0]
v2_frame=tric_basis[1]+tric_basis[2]
df_cut_ori.loc[:,['x','y','z']]=rotate.rotate_twice_linker(df_cut_ori,center,v1_file,v1_frame,v2_file,v2_frame)
df_cut_ori.insert(3, "note", df_cut_ori['Note'], True)
df_cut_ori.to_csv('1rotatecut_lib.txt', sep="\t", header=None, index=False)
output('1rotatecut_lib', outgro=False,outpdb=True,outxyz=True)