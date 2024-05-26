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
        ) = x_num, y_num, z_num, x_scalar, y_scalar, z_scalar
        self.dx_value, self.dy_value, self.dz_value = 1, 1, 1
        self.dx, self.dy, self.dz = dx, dy, dz
        self.tric_basis = tric_basis
        self.axis_lib, self.local_pdb = node_axis_lib, node_local_pdb

    def get_points(self):
        points = pre_process.points_generator(
            self.x_num,
            self.y_num,
            self.z_num,
            self.dx_value,
            self.dy_value,
            self.dz_value,
            self.dx,
            self.dy,
            self.dz,
        )
        # Amap needs to decribe all A in single unit box
        A_map_0 = pre_process.points_generator(
            self.x_num,
            self.y_num,
            self.z_num,
            2 * self.dx_value,
            2 * self.dy_value,
            2 * self.dz_value,
            self.dx,
            self.dy,
            self.dz,
        )
        A_map_1, A_map_2, A_map_3 = A_map_0 + np.array([1, 1, 0]), A_map_0 + np.array([1, 0, 1]), A_map_0 + np.array([0, 1, 1])
        A_map = np.concatenate((A_map_0, A_map_1,A_map_2,A_map_3), axis=0)
        B_map_0, B_map_1, B_map_2 = A_map + np.array([1, 0, 0]), A_map + np.array([0, 1, 0]), A_map + np.array([0, 0, 1])
        B_map = np.concatenate((B_map_0, B_map_1,B_map_2), axis=0)
        A_map, B_map = np.unique(A_map, axis=0), np.unique(B_map, axis=0)

        group_A = pre_process.find_overlapped_3D_array(A_map, points)
        group_B = pre_process.find_overlapped_3D_array(B_map, points)
        # find linker positions by points groups

        # points_c_linker_dx = pre_process.find_overlapped_3D_array(points+np.array([1,0,0]),points)-0.5*np.array([1,0,0])
        points_c_linker_dy = pre_process.find_overlapped_3D_array(
            group_A + np.array([1, 1, 0]), group_A
        ) - 0.5 * np.array([1, 1, 0])
        #points_c_linker_dz1 = pre_process.find_overlapped_3D_array(
        #     group_B + np.array([1, 0, 1]), group_A
        #) - 0.5 * np.array([1, 0, 1])
        points_c_linker_dz = pre_process.find_overlapped_3D_array(
             group_B + np.array([1, 0, 1]), group_B
        ) - 0.5 * np.array([1, 0, 1])
        #points_c_linker_dz = np.vstack((points_c_linker_dz1,points_c_linker_dz2))
        #points_c_linker_dz = np.unique(points_c_linker_dz,axis=0)

        points_c = np.vstack([points_c_linker_dy, points_c_linker_dz])
        carte_points = pre_process.zoom_points(
            points, self.x_scalar, self.y_scalar, self.z_scalar
        )
        # group_A = pre_process.zoom_points(group_A,self.x_scalar,self.y_scalar,self.z_scalar)
        # group_B = pre_process.zoom_points(group_B,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dx = pre_process.zoom_points(points_c_linker_dx,self.x_scalar,self.y_scalar,self.z_scalar)
        carte_points_c_dy = pre_process.zoom_points(
            points_c_linker_dy, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dz = pre_process.zoom_points(
            points_c_linker_dz, self.x_scalar, self.y_scalar, self.z_scalar
        )
        # carte_points_c_dxy = pre_process.zoom_points(points_c_linker_dxy,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dxz = pre_process.zoom_points(points_c_linker_dxz,self.x_scalar,self.y_scalar,self.z_scalar)
        # carte_points_c_dyz = pre_process.zoom_points(points_c_linker_dyz,self.x_scalar,self.y_scalar,self.z_scalar)

        carte_points_c = []
        for i in [carte_points_c_dy, carte_points_c_dz]:
            carte_points_c.append(i)

        points_c = pre_process.find_overlapped_3D_array(
            group_A + np.array([1, 1, 1]), points
        )
        points = pre_process.zoom_points(
            points, self.x_scalar, self.y_scalar, self.z_scalar
        )
        group_A = pre_process.zoom_points(
            group_A, self.x_scalar, self.y_scalar, self.z_scalar
        )
        group_B = pre_process.zoom_points(
            group_B, self.x_scalar, self.y_scalar, self.z_scalar
        )
        points_c = pre_process.zoom_points(
            points_c, self.x_scalar, self.y_scalar, self.z_scalar
        )
        print(
            "\npointsA number:  "
            + str(group_A.shape[0])
            + "\npointsB number:  "
            + str(group_B.shape[0])
            + "\nallpoints number:  "
            + str(points.shape[0])
        )
        return carte_points, group_A, group_B, carte_points_c

    def node_learn_from_template(self):
        solution_1_2, arr_1_2, solution_1_3, arr_1_3 = read.read_axis_from_lib(
            self.axis_lib
        )
        local_pdb = read.pdb(self.local_pdb)
        axis1 = self.dx  # ATTENTION
        axis2 = pre_process.get_axis2(solution_1_2, arr_1_2, solution_1_3, arr_1_3)
        # axis3 = np.cross(axis1, axis2)

        point_Al = local_pdb.loc[0, ["x", "y", "z"]].to_numpy()
        p1, p2, p3 = (
            local_pdb.loc[1, ["x", "y", "z"]].to_numpy() - point_Al,
            local_pdb.loc[2, ["x", "y", "z"]].to_numpy() - point_Al,
            local_pdb.loc[3, ["x", "y", "z"]].to_numpy() - point_Al,
        )
        p1, p2, p3 = normalize_vector(p1), normalize_vector(p2), normalize_vector(p3)
        arr = np.vstack((p1, p2, p3))
        V1, V2 = np.dot(solution_1_2, arr), np.dot(solution_1_3, arr)
        V1, V2 = normalize_vector(V1), normalize_vector(V2)

        Al_node = (
            local_pdb.loc[:, ["x", "y", "z"]].to_numpy() - point_Al
        )  # MOVE center (Al this case) to (0,0,0)
        q1 = rotate.calculate_q_rotation_with_vectors(V1, axis1)
        q_V2 = quaternion.from_vector_part(V2)
        new_q_V2 = q1 * q_V2
        new_V2 = quaternion.as_vector_part(new_q_V2)

        q2 = rotate.calculate_q_rotation_with_vectors(new_V2, axis2)
        # q3 = quaternion.from_float_array([0,0,0,-1])
        # dy dz rotate pi
        # q3 = rotate.calculate_q_rotation_with_axis_degree(axis2,np.pi)*rotate.calculate_q_rotation_with_axis_degree(axis3,np.pi)
        q_A = q2 * q1
        # q_B = q3*q2*q1

        new_node_A = rotate.get_rotated_array(Al_node, q_A)
        # new_node_B = rotate.get_rotated_array(Al_node,q_B)

        a, b, c = self.tric_basis[0], self.tric_basis[1], self.tric_basis[2]
        oh_direction = b + c
        old_oh_direction = axis2
        angle = rotate.calculate_angle_rad(a, old_oh_direction, oh_direction)
        q_nodeA = rotate.calculate_q_rotation_with_axis_degree(a, angle)
        new_node_A = rotate.get_rotated_array(new_node_A, q_nodeA)
        ## q_nodeB = rotate.calculate_q_rotation_with_axis_degree(a, angle + 0.5 * np.pi)
        yz_mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        new_node_B = np.dot(new_node_A, yz_mirror_matrix)

        return new_node_A, new_node_B,q_nodeA

