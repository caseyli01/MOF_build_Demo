import datetime
import numpy as np
import py3Dmol as p3d
import os
from functions.fetch import fetch
from functions.output import output 
from functions.clean import clean 
from functions.save import save 
from functions.rotate import rotate
from UIO66.UIO66frame import Frame
from UIO66.UIO66node import get_node
from UIO66.UIO66linker import get_linker
from UIO66.UIO66termination import get_term


# -----------------------------------------------------------------------------------------------------------#

startTime = datetime.datetime.now()
# lib file
libpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')

any_tdx, any_tdy, any_tdz = (
    np.array([1, 1, 0]),
    np.array([0, 1, 1]),
    np.array([1, 0, 1])

    
)  # face center cubic for uio66 fcb



class UIO66:
    def __init__(
        self,
        linker_pdb,
        O1_index,
        O2_index,
        O3_index,
        x_num,
        y_num,
        z_num,
        x_scalar,
        y_scalar,
        z_scalar,
    ):
        self.x_num, self.y_num, self.z_num = x_num, y_num, z_num
        self.x_scalar, self.y_scalar, self.z_scalar = x_scalar, y_scalar, z_scalar
        self.any_tdx, self.any_tdy, self.any_tdz = any_tdx, any_tdy, any_tdz
        self.linker_pdb = linker_pdb
        self.node_axis_lib, self.node_local_pdb, self.cut_lib_pdb = libpath+'/UIO66_lib_axisZr.lib',libpath+'/UIO66_lib_nodeZr.lib',libpath+'/UIO66_lib_cut.lib' 

        self.O1_index, self.O2_index, self.O3_index = (
            O1_index,
            O2_index,
            O3_index,
        )






    def buildall(self,outnode, outcut,outlinker,  outall):
        self.outnode = outnode
        self.outlinker = outlinker
        self.outcut = outcut
        self.outall = outall

        """coordinate"""
        # cartesian
        dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        carte_basis = np.vstack([dx, dy, dz])
        # for uio66 fcb
        tric_basis = rotate.coordinate_transfer(any_tdx, any_tdy, any_tdz, carte_basis)

        """frame"""
        frame = Frame(
            self.x_num, self.y_num, self.z_num, self.x_scalar, self.y_scalar, self.z_scalar, dx, dy, dz, tric_basis, self.node_axis_lib, self.node_local_pdb
        )
        carte_points, carte_points_c = frame.get_points()
        tric_points = np.dot(
            carte_points, tric_basis
        )  # rotate.coordinate_transfer(any_tdx,any_tdy,any_tdz,carte_points)
        tric_points_c = []
        for i in range(6):  # 6 axis for 12 edge
            tric_points_c.append(np.dot(carte_points_c[i], tric_basis))

        """node"""
        new_node_A = frame.node_learn_from_template()
        df_node = get_node(self.node_local_pdb, tric_points, new_node_A, outnode)
        output(outnode, outgro=False, outpdb=False, outxyz=True)

        """linker"""
        df_linker = get_linker(
            tric_basis, new_node_A, self.linker_pdb, self.O1_index, self.O2_index, self.O3_index, tric_points_c, outlinker
        )
        ##NOTE:O1->O2//short edge O-O ,O1->O3//long edge O-O
        output(outlinker, outgro=False, outpdb=False, outxyz=True)

        """cut"""
        df_cut = get_term(
            self.cut_lib_pdb, self.x_num, self.y_num,self.z_num, self.x_scalar, self.y_scalar, self.z_scalar, tric_basis, carte_points, tric_points, outcut
        )
        output(outcut, outgro=False, outpdb=False, outxyz=True)
        """all"""
        residues, res_count = fetch.reslist_resnum(
            self.node_local_pdb,  self.cut_lib_pdb,self.linker_pdb,1,12,1,linker_order=3
        )
        df_all = save.all(df_node, df_cut,df_linker, outall, residues, res_count)

        output(outall, outgro=True, outpdb=True, outxyz=True)

        endTime = datetime.datetime.now()
        print("\n" + "Time cost (s):   " + str(endTime - startTime))
        clean.txt()

    def view(self,xyzname):
            viewer = p3d.view(width=600, height=600)
            viewer.addModelsAsFrames(open(xyzname+'.xyz', "r").read(), "xyz", {"keepH": True})
            viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
            viewer.zoomTo()
            viewer.show()
