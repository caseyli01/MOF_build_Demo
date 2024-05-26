import datetime
import numpy as np
import py3Dmol as p3d
import os
from functions.fetch import fetch
from MIL53.MIL53frame import Frame
from functions.output import output 
from functions.clean import clean 
from functions.save import save 
from functions.rotate import rotate
from MIL53.MIL53node import get_node
from MIL53.MIL53linker import get_linker
from MIL53.MIL53termination import get_term


# ---------------------------------------------------------------------------------------------------------------#
startTime = datetime.datetime.now()

libpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lib')
class MIL53:
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
        any_tdx,
        any_tdy,
        any_tdz,
    ):
        self.x_num, self.y_num, self.z_num = x_num, y_num, z_num
        self.x_scalar, self.y_scalar, self.z_scalar = x_scalar, y_scalar, z_scalar
        self.any_tdx, self.any_tdy, self.any_tdz = any_tdx, any_tdy, any_tdz
        self.linker_pdb = linker_pdb
        self.node_axis_lib, self.node_local_pdb, self.cut_lib_pdb = libpath+'/MIL53_lib_axisAl.lib',libpath+'/MIL53_lib_nodeAl.lib',libpath+'/MIL53_lib_cut.lib' 

        self.O1_index, self.O2_index, self.O3_index = (
            O1_index,
            O2_index,
            O3_index,
        )

    def buildall(self, outnode, outcut,outlinker,  outall):
        self.outnode = outnode
        self.outlinker = outlinker
        self.outcut = outcut
        self.outall = outall
        """coordinate"""
        # cartesian
        dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        carte_basis = np.vstack([dx, dy, dz])
        tric_basis = rotate.coordinate_transfer(
            self.any_tdx, self.any_tdy, self.any_tdz, carte_basis
        )

        """frame"""
        frame = Frame(
            self.x_num,
            self.y_num,
            self.z_num,
            self.x_scalar,
            self.y_scalar,
            self.z_scalar,
            dx,
            dy,
            dz,
            tric_basis,
            self.node_axis_lib,
            self.node_local_pdb,
        )

        carte_points, group_A, group_B, carte_points_c = frame.get_points()
        new_node_A, new_node_B, q_nodeA = frame.node_learn_from_template()

        tric_points = np.dot(carte_points, tric_basis)
        tric_points_c = []
        for i in range(2):  # 2 axis for 4 edge
            tric_points_c.append(np.dot(carte_points_c[i], tric_basis))

        """node"""
        df_node = get_node(
            self.node_local_pdb,
            group_A,
            group_B,
            tric_basis,
            new_node_A,
            new_node_B,
            outnode,
        )
        output(outnode, outgro=False, outpdb=False, outxyz=True)

        """linker"""
        df_linker = get_linker(
            tric_basis,
            self.linker_pdb,
            self.O1_index,
            self.O2_index,
            self.O3_index,
            tric_points_c,
            outlinker,
        )
        ##NOTE:O1->O2//short edge O-O ,O1->O3//long edge O-O
        output(outlinker, outgro=False, outpdb=False, outxyz=True)

        """cut"""
        df_cut = get_term(
            self.cut_lib_pdb,
            self.x_num,
            self.y_num,
            self.z_num,
            self.x_scalar,
            self.y_scalar,
            self.z_scalar,
            tric_basis,
            q_nodeA,
            group_A,
            group_B,
            outcut,
        )
        output(outcut, outgro=False, outpdb=False, outxyz=True)

        """all"""
        residues, res_count = fetch.reslist_resnum( 
            self.node_local_pdb, self.cut_lib_pdb,self.linker_pdb, 1,2,1,linker_order=3 #cut.lib has two repeated cut
        )
        df_all = save.all(df_node,df_cut,  df_linker, outall, residues, res_count)

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
