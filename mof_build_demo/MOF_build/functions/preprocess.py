import numpy as np
import quaternion
from .rotate import rotate
from .normv import normalize_vector


class pre_process:
    """pre-process"""

    def get_axis2(solution_1_2, arr_1_2, solution_1_3, arr_1_3):
        axis1 = np.dot(solution_1_2, arr_1_2)
        axis2 = np.dot(solution_1_3, arr_1_3)
        q_axis = rotate.calculate_q_rotation_with_vectors(axis1, axis2)
        dx = np.array([1, 0, 0])
        axis0 = quaternion.from_vector_part(dx)
        new_axis = q_axis * axis0
        new_axis_vector = quaternion.as_vector_part(new_axis)
        # print(new_axis_vector)
        return new_axis_vector

    def points_generator(x_num, y_num, z_num, dx_value, dy_value, dz_value, dx, dy, dz):
        """this function is to generate a group of 3d points(unit=1) defined by user for further grouping points"""
        unit_dx = dx_value * dx  # dx_value works as a scalar
        unit_dy = dy_value * dy
        unit_dz = dz_value * dz
        # add x layer
        points = np.array([0, 0, 0])
        for i in range(0, x_num + 1):
            points = np.vstack((points, i * unit_dx))
        # add y layer
        points_x = points
        for i in range(0, y_num + 1):
            points = np.vstack((points, points_x + i * unit_dy))
        # add z layer
        points_xy = points
        for i in range(0, z_num + 1):
            points = np.vstack((points, points_xy + i * unit_dz))
        points = np.unique(points, axis=0)
        return points

    def zoom_points(points, x_scalar, y_scalar, z_scalar):
        points = np.array(points, dtype=float)
        result = points * np.array([x_scalar, y_scalar, z_scalar])
        return result

    def find_overlapped_3D_array(array1, array2):
        set1 = set(map(tuple, array1.reshape(-1, array1.shape[-1])))
        set2 = set(map(tuple, array2.reshape(-1, array2.shape[-1])))
        # Find intersection of sets
        overlapped_elements = set1.intersection(set2)
        # Convert back to numpy array
        overlapped_array = np.array(list(overlapped_elements)).reshape(
            -1, array1.shape[-1]
        )
        return overlapped_array

    def find_solution(pAl1, pAl1_1, pAl1_2, pAl1_3,vAl_Al):
        Al1_Al2 = vAl_Al
        vAl1_Al2 = normalize_vector(Al1_Al2)
        v12_1, v12_2, v12_3 = pAl1_1 - pAl1, pAl1_2 - pAl1, pAl1_3 - pAl1
        v12_1, v12_2, v12_3 = (
            normalize_vector(v12_1),
            normalize_vector(v12_2),
            normalize_vector(v12_3),
        )
        arr_1_2 = np.vstack((v12_1, v12_2, v12_3))
        arr_1_2 = arr_1_2.astype(np.float64)
        vAl1_Al2 = vAl1_Al2.astype(np.float64)

        solution_1_2 = np.dot(vAl1_Al2, np.linalg.inv(arr_1_2))
        return solution_1_2, arr_1_2

