import numpy as np
import quaternion
from .normv import normalize_vector

class rotate:
    """rotate"""

    def calculate_q_rotation_with_vectors(p1, p2):
        p1 = normalize_vector(p1)
        p2 = normalize_vector(p2)
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        if abs(np.dot(p1, p2)) > 0.9999:
            q = [1, 0, 0, 0]
        else:
            q_xyz = np.cross(p1, p2)
            q_w = np.array([1 + np.dot(p1, p2)], dtype=float)
            q = np.concatenate([q_w, q_xyz])
        r_q = quaternion.from_float_array(q)
        return r_q

    def get_rotated_array(arr, q):
        q_arr = quaternion.from_vector_part(arr)
        rotated_q_arr = q * q_arr * q.inverse()
        rotated_arr = quaternion.as_vector_part(rotated_q_arr)
        return rotated_arr

    def calculate_angle_rad(axis, p1, p2):
        """this function is to calculte the rotation angle of specifc rotation axis
        and the two vectors is before and after"""
        axis = normalize_vector(axis)
        a_square = np.linalg.norm(p1) * np.linalg.norm(p1) - np.dot(p1, axis) * np.dot(
            p1, axis
        )
        b_square = np.linalg.norm(p2) * np.linalg.norm(p2) - np.dot(p2, axis) * np.dot(
            p2, axis
        )
        c_square = np.linalg.norm(p2 - p1) * np.linalg.norm(p2 - p1)
        a, b = np.sqrt(a_square), np.sqrt(b_square)
        if abs(a * b) < 0.00001:
            cos_theta = 1
        else:
            cos_theta = np.clip((a_square + b_square - c_square) / (2 * a * b), -1, 1)
        theta_rad = np.arccos(cos_theta)
        return theta_rad

    def calculate_q_rotation_with_axis_degree(
        axis, theta
    ):  # axis is HE---HE ,theta from O1--AXIS--O1'
        """
        this function is to get quaternion form rotation operator of along specific rotation axis,
        this is for the second rotation along the self-axis to fix the posture of the object
        """
        w = theta / 2
        s = np.sin(w)
        q_real = np.array([np.cos(w)])
        q_ijk = s * axis
        q_r = np.concatenate([q_real, q_ijk])
        q_r = quaternion.from_float_array(q_r)
        return q_r

    def rotate_twice_linker(
        df_input, beginning_point, v1_file, v1_frame, v2_file, v2_frame
    ):
        """
        we need to rotate an object twice to make sure the position and posture is right,
        the first is rotation from vector to vector directly,
        the second is rotation along self-axis, angle is calculated in this step which introduces precision loss
        """
        arr = (
            df_input.loc[:, ["x", "y", "z"]].to_numpy() - beginning_point
        )  # MOVE center (Al this case) to (0,0,0)
        q0 = rotate.calculate_q_rotation_with_vectors(v1_file, v1_frame)
        q1=q0
        #if (q0 == quaternion.from_float_array([1,0,0,0]) and (np.dot(v1_file,v1_frame)<0)):
        #        q1 = quaternion.from_float_array([-1,0,0,0])
        #else:
        #        q1 = q0
        q_V2 = quaternion.from_vector_part(v2_file)
        new_q_V2 = q1 * q_V2
        new_V2_file = quaternion.as_vector_part(new_q_V2)
        angle = rotate.calculate_angle_rad(v1_frame, new_V2_file, v2_frame)
        q2 = rotate.calculate_q_rotation_with_axis_degree(v1_frame, angle)
        # q2 = rotate.calculate_q_rotation_with_vectors(new_V2_file,v2_frame)
        q_rotate = q2 * q1
        new_array = rotate.get_rotated_array(arr, q_rotate)
        return new_array

    def coordinate_transfer(any_tdx, any_tdy, any_tdz, points):
        """
        for cartesian coordinate int points, it is much easier to get the unique and overlapped points
        and we apply basis transfer of coordinate to generate non-Cartesian dx dy dz periodic system
        we will make the new basis[0] stick to [1,0,0] for further operations
        """
        dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
        carte = np.vstack([dx, dy, dz])
        tdx, tdy, tdz = (
            normalize_vector(any_tdx),
            normalize_vector(any_tdy),
            normalize_vector(any_tdz),
        )
        tric = np.vstack([tdx, tdy, tdz])
        for i in range(3):
            v1 = carte[i, :]  # dx
            v2 = tric[i, :]
            if np.dot(v1, v2) < 0.9999:
                break
        # axis = carte[2,:] #dz
        axis = normalize_vector(np.cross(v1, v2))
        theta = rotate.calculate_angle_rad(axis, v2, v1)
        r = rotate.calculate_q_rotation_with_axis_degree(axis, theta)
        # r = rotate.calculate_q_rotation_with_vectors(v1,v2)
        r_m = quaternion.as_rotation_matrix(r)
        # new_tric_basis=np.round(np.dot(tric,r_m),5)
        new_tric_basis = np.dot(tric, r_m)
        new_tric_points = np.dot(points, new_tric_basis)
        return new_tric_points
