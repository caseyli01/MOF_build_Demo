import numpy as np
import quaternion
from MOF_build.functions.rotate  import rotate
from MOF_build.functions.read import read
from MOF_build.functions.preprocess import  pre_process
from MOF_build.functions.normv import normalize_vector


class Frame:
    def __init__(
        self,
        x_num,
        y_num,
        z_num,
        x_scalar,
        y_scalar,
        z_scalar,
        dx,
        dy,
        dz,
        tric_basis,
        node_axis_lib,
        node_local_pdb,
    ):
        (
            self.x_num,
            self.y_num,
            self.z_num,
            self.x_scalar,
            self.y_scalar,
            self.z_scalar,
        ) = x_num, y_num, z_num, 2*x_scalar,2*y_scalar, 2*z_scalar #2* is because the linker edge length is 0.5*of cell edge length
        self.dx_value, self.dy_value, self.dz_value = 1, 1, 1
        self.dx, self.dy, self.dz = dx, dy, dz
        self.tric_basis = tric_basis
        self.axis_lib, self.local_pdb = node_axis_lib, node_local_pdb

    def get_points(self):
        points = pre_process.points_generator(
            self.x_num,
            self.y_num,
            self.z_num,
            0.5*self.dx_value,
            self.dy_value,
            self.dz_value,
            self.dx,
            self.dy,
            self.dz,
        )

        group_A = pre_process.find_overlapped_3D_array(
            points + np.array([0, 0, 1]), points
        ) - 0.5 * np.array([0, 0, 1])
        
        group_B = pre_process.find_overlapped_3D_array(
            points + np.array([0, 1, 1]), points
        ) - 0.5 * np.array([0, 1, 1])

        group_C = pre_process.find_overlapped_3D_array(
            points + np.array([0, 1, 0]), points
        ) - 0.5 * np.array([0, 1, 0])

        linker_c_AACC1 = pre_process.find_overlapped_3D_array(
            group_A + 0.5*np.array([1, 1, -1]), group_C
        ) - 0.25 * np.array([1, 1, -1])
        linker_c_AACC2 = pre_process.find_overlapped_3D_array(
            group_A - 0.5*np.array([1, 1, -1]), group_C
        ) + 0.25 * np.array([1, 1, -1])
        linker_c_AACC = np.vstack((linker_c_AACC1,linker_c_AACC2))

        linker_c_AABB1 = pre_process.find_overlapped_3D_array(
            group_A + 0.5*np.array([1, 1, 0]), group_B
        ) - 0.25 * np.array([1, 1, 0])
        linker_c_AABB2 = pre_process.find_overlapped_3D_array(
            group_A - 0.5*np.array([1, 1, 0]), group_B
        ) + 0.25 * np.array([1, 1, 0])
        linker_c_AABB = np.vstack((linker_c_AABB1,linker_c_AABB2))

        linker_c_BBCC1 = pre_process.find_overlapped_3D_array(
            group_C + 0.5*np.array([1, 0, 1]), group_B
        ) - 0.25 * np.array([1, 0, 1])
        linker_c_BBCC2 = pre_process.find_overlapped_3D_array(
            group_C - 0.5*np.array([1, 0, 1]), group_B
        ) + 0.25 * np.array([1, 0, 1])
        linker_c_BBCC = np.vstack((linker_c_BBCC1,linker_c_BBCC2))


        carte_points = pre_process.zoom_points(
            points, self.x_scalar, self.y_scalar, self.z_scalar
        )
        # group_A = pre_process.zoom_points(group_A,self.x_scalar,self.y_scalar,self.z_scalar)
        # group_B = pre_process.zoom_points(group_B,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dx = pre_process.zoom_points(points_c_linker_dx,self.x_scalar,self.y_scalar,self.z_scalar)
        carte_points_c_AACC1 = pre_process.zoom_points(
            linker_c_AACC1, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_AABB1 = pre_process.zoom_points(
            linker_c_AABB1, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_BBCC1 = pre_process.zoom_points(
            linker_c_BBCC1, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_AACC2 = pre_process.zoom_points(
            linker_c_AACC2, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_AABB2 = pre_process.zoom_points(
            linker_c_AABB2, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_BBCC2 = pre_process.zoom_points(
            linker_c_BBCC2, self.x_scalar, self.y_scalar, self.z_scalar
        )
     
        # carte_points_c_dxy = pre_process.zoom_points(points_c_linker_dxy,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dxz = pre_process.zoom_points(points_c_linker_dxz,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dyz = pre_process.zoom_points(points_c_linker_dyz,self.x_scalar,self.y_scalar,self.z_scalar)
        
        group_A = pre_process.zoom_points(
            group_A, self.x_scalar, self.y_scalar, self.z_scalar
        )
        group_B = pre_process.zoom_points(
            group_B, self.x_scalar, self.y_scalar, self.z_scalar
        )
        group_C = pre_process.zoom_points(
            group_C, self.x_scalar, self.y_scalar, self.z_scalar
        )

        carte_points_c = []
        carte_points_c.append(carte_points_c_AABB1)
        carte_points_c.append(carte_points_c_BBCC2)
        carte_points_c.append(carte_points_c_AACC1)
        carte_points_c.append(carte_points_c_AABB2)
        carte_points_c.append(carte_points_c_BBCC1)
        carte_points_c.append(carte_points_c_AACC2)

        carte_points_ABC=np.vstack((group_A,group_B,group_C))
        
        print(
            "\npointsA number:  "
            + str(group_A.shape[0])
            + "\npointsB number:  "
            + str(group_B.shape[0])
            + "\npointsC number:  "
            + str(group_C.shape[0])
            + "\nallpoints number:  "
            + str(group_A.shape[0]+group_B.shape[0]+group_C.shape[0])
        )
        return  carte_points_ABC, group_A, group_B, group_C, carte_points_c

    def node_learn_from_template(self):
        solution_1_2, arr_1_2, solution_1_3, arr_1_3 = read.read_axis_from_lib(
            self.axis_lib
        )
        local_pdb = read.pdb(self.local_pdb)
        #axis1 = self.dx  # ATTENTION 
        #axis2 = pre_process.get_axis2(solution_1_2, arr_1_2, solution_1_3, arr_1_3)
        # axis3 = np.cross(axis1, axis2)

        
        p1,p2,p3,p4 = (local_pdb.loc[61, ['x','y','z']].to_numpy(),
                local_pdb.loc[52, ['x','y','z']].to_numpy(),
                local_pdb.loc[34, ['x','y','z']].to_numpy(),
                local_pdb.loc[16, ['x','y','z']].to_numpy(),
                ) 
        V1=p2-p1
        V2=p3-p4
        center_point=0.5*(p2+p1)
        V1, V2 = normalize_vector(V1), normalize_vector(V2)
        

        
        v1_file , v2_file = V1, V2
        v1_frame = self.tric_basis[1]+self.tric_basis[2] # ATTENTION  defined by lib file making process
        v2_frame = self.tric_basis[0] # ATTENTION 
        
        new_node_B = rotate.rotate_twice_linker(local_pdb,center_point,v2_file,v2_frame,v1_file,v1_frame)
        q_rotate_local_node = rotate.get_q_rotate_twice_linker(local_pdb,center_point,v2_file,v2_frame,v1_file,v1_frame)
        angle = -1*np.pi/3
        q_rotate = rotate.calculate_q_rotation_with_axis_degree(self.tric_basis[0], angle)
        
        #new_node_A = rotate.get_rotated_array(new_node_A, q_nodeA)
        ## q_nodeB = rotate.calculate_q_rotation_with_axis_degree(a, angle + 0.5 * np.pi)
        #yz_mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        #new_node_B = np.dot(new_node_A, yz_mirror_matrix)
        new_node_A = rotate.get_rotated_array(new_node_B, q_rotate)
        new_node_C = rotate.get_rotated_array(new_node_A, q_rotate)
        
        #q_rotate_local_node to rotate cut as well
        return new_node_A, new_node_B, new_node_C, q_rotate_local_node #q_rotate_local_node to rotate cut as well

if __name__ == "__main__":
    print('test')
    
    any_tdx, any_tdy, any_tdz = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 1]))
    dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    carte_basis = np.vstack([dx, dy, dz])
        # for uio66 fcb
    tric_basis = rotate.coordinate_transfer(any_tdx, any_tdy, any_tdz, carte_basis)
    x_num, y_num, z_num, x_scalar, y_scalar, z_scalar = 1,1,1,1,1,1
    libpath='D:\OneDrive - KTH\data\code\MOF_build_Demo\mof_build_demo\MOF_build\lib'
    node_axis_lib, node_local_pdb, cut_lib_pdb = libpath+'/PCN222_lib_axisZr.lib',libpath+'/PCN222_lib_nodeZr.lib',libpath+'/PCN222_lib_cut.lib' 
    frame = Frame(
            x_num, y_num, z_num, x_scalar, y_scalar, z_scalar, dx, dy, dz, tric_basis,node_axis_lib, node_local_pdb
        )
    carte_points, group_A, group_B, group_C, carte_points_c = frame.get_points()
    new_node_B, new_node_A,  new_node_C = frame.node_learn_from_template()
    print(group_A,group_B,group_C)
