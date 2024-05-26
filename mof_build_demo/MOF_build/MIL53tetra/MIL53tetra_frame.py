import numpy as np
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
            self.dz_value,
            self.dx,
            self.dy,
            self.dz,
        )
        A_map_1 = A_map_0 + np.array([1, 0, 0]) + np.array([0, 1, 0])
        A_map = np.concatenate((A_map_0, A_map_1), axis=0)
        B_map_0, B_map_1 = A_map + np.array([1, 0, 0]), A_map + np.array([0, 1, 0])
        B_map = np.concatenate((B_map_0, B_map_1), axis=0)
        A_map, B_map = np.unique(A_map, axis=0), np.unique(B_map, axis=0)
        group_A = pre_process.find_overlapped_3D_array(A_map, points)
        group_B = pre_process.find_overlapped_3D_array(B_map, points)
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
        return points, group_A, group_B, points_c

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

        # Al_node = (
        #    local_pdb.loc[:, ["x", "y", "z"]].to_numpy() - point_Al
        # )  # MOVE center (Al this case) to (0,0,0)
        # q1 = rotate.calculate_q_rotation_with_vectors(V1, axis1)
        # q_V2 = quaternion.from_vector_part(V2)
        # new_q_V2 = q1 * q_V2
        # new_V2 = quaternion.as_vector_part(new_q_V2)

        # q2 = rotate.calculate_q_rotation_with_vectors(new_V2, axis2)
        # q3 = quaternion.from_float_array([0,0,0,-1])
        # dy dz rotate pi
        # q3 = rotate.calculate_q_rotation_with_axis_degree(
        #    axis2, np.pi
        # ) * rotate.calculate_q_rotation_with_axis_degree(axis3, np.pi)
        # q_A = q2 * q1

        new_node_A = rotate.rotate_twice_linker(
            local_pdb, point_Al, V1, axis1, V1, axis1
        )
        yz_mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        new_node_B = np.dot(new_node_A, yz_mirror_matrix)
        return new_node_A, new_node_B
