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
        self.tdx, self.tdy, self.tdz = (
            tric_basis[0, :],
            tric_basis[1, :],
            tric_basis[2, :],
        )
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

        points_c_linker_dx = pre_process.find_overlapped_3D_array(
            points + np.array([1, 0, 0]), points
        ) - 0.5 * np.array([1, 0, 0])
        points_c_linker_dy = pre_process.find_overlapped_3D_array(
            points + np.array([0, 1, 0]), points
        ) - 0.5 * np.array([0, 1, 0])
        points_c_linker_dz = pre_process.find_overlapped_3D_array(
            points + np.array([0, 0, 1]), points
        ) - 0.5 * np.array([0, 0, 1])
        points_c_linker_dxy = pre_process.find_overlapped_3D_array(
            points + np.array([1, 1, 0]), points
        ) - 0.5 * np.array([1, 1, 0])
        points_c_linker_dxz = pre_process.find_overlapped_3D_array(
            points + np.array([1, 0, 1]), points
        ) - 0.5 * np.array([1, 0, 1])
        points_c_linker_dyz = pre_process.find_overlapped_3D_array(
            points + np.array([0, 1, 1]), points
        ) - 0.5 * np.array([0, 1, 1])

        points_c = np.vstack(
            [
                points_c_linker_dx,
                points_c_linker_dy,
                points_c_linker_dz,
                points_c_linker_dxy,
                points_c_linker_dxz,
                points_c_linker_dyz,
            ]
        )
        carte_points = pre_process.zoom_points(
            points, self.x_scalar, self.y_scalar, self.z_scalar
        )
        # group_A = pre_process.zoom_points(group_A,self.x_scalar,self.y_scalar,self.z_scalar)
        # group_B = pre_process.zoom_points(group_B,self.x_scalar,self.y_scalar,self.z_scalar)
        carte_points_c_dx = pre_process.zoom_points(
            points_c_linker_dx, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dy = pre_process.zoom_points(
            points_c_linker_dy, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dz = pre_process.zoom_points(
            points_c_linker_dz, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dxy = pre_process.zoom_points(
            points_c_linker_dxy, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dxz = pre_process.zoom_points(
            points_c_linker_dxz, self.x_scalar, self.y_scalar, self.z_scalar
        )
        carte_points_c_dyz = pre_process.zoom_points(
            points_c_linker_dyz, self.x_scalar, self.y_scalar, self.z_scalar
        )

        carte_points_c = []
        for i in [
            carte_points_c_dx,
            carte_points_c_dy,
            carte_points_c_dz,
            carte_points_c_dxy,
            carte_points_c_dxz,
            carte_points_c_dyz,
        ]:
            carte_points_c.append(i)

        print(
            "\npoints number:  "
            + str(points.shape[0])
            + "\nlinkers number:  "
            + str(points_c.shape[0])
        )
        return carte_points, carte_points_c

    def node_learn_from_template(self):
        solution_1_2, arr_1_2, solution_1_3, arr_1_3 = read.read_axis_from_lib(
            self.axis_lib
        )
        local_pdb = read.pdb(self.local_pdb)
        axis1 = self.tdx  # ATTENTION
        axis2 = self.tdy
        # pre_process.get_axis2(solution_1_2,arr_1_2,solution_1_3,arr_1_3)
        # axis3 = np.cross(axis1,axis2) #if need bottom-up rotation
        point_center = (
            local_pdb.loc[0, ["x", "y", "z"]].to_numpy()
            + local_pdb.loc[9, ["x", "y", "z"]].to_numpy()
        )
        p1, p2, p3 = (
            local_pdb.loc[0, ["x", "y", "z"]].to_numpy() - point_center,
            local_pdb.loc[36, ["x", "y", "z"]].to_numpy() - point_center,
            local_pdb.loc[45, ["x", "y", "z"]].to_numpy() - point_center,
        )
        p1, p2, p3 = normalize_vector(p1), normalize_vector(p2), normalize_vector(p3)
        arr = np.vstack((p1, p2, p3))
        V1, V2 = np.dot(solution_1_2, arr), np.dot(solution_1_3, arr)
        V1, V2 = normalize_vector(V1), normalize_vector(V2)
        # rotate for more, twice is minimum
        # node = local_pdb.loc[:,['x','y','z']].to_numpy() - point_center  #MOVE center (Al this case) to (0,0,0)
        new_node_A = rotate.rotate_twice_linker(
            local_pdb, point_center, V1, axis1, V2, axis2
        )
        # q3 = rotate.calculate_q_rotation_with_axis_degree(axis2,np.pi)*rotate.calculate_q_rotation_with_axis_degree(axis3,np.pi)

        return new_node_A

