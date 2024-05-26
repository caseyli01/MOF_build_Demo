import numpy as np
import pandas as pd
import quaternion
import re
import os
import glob


def normalize_vector(v):
    norm_v = v / np.linalg.norm(v)
    return norm_v


class clean:
    def gro():
        files = glob.glob("*.gro")
        for f in files:
            os.remove(f)

    def xyz():
        files = glob.glob("*.xyz")
        for f in files:
            os.remove(f)

    def pdb():
        files = glob.glob("*.pdb")
        for f in files:
            os.remove(f)

    def csv():
        files = glob.glob("*.csv")
        for f in files:
            os.remove(f)

    def txt():
        files = glob.glob("*.txt")
        for f in files:
            os.remove(f)


class read:
    def pdb(filename):
        inputfile = str(filename)
        outputfile = re.sub(r"\..*$", "", inputfile)
        with open(inputfile, "r") as fp:
            content = fp.readlines()
            linesnumber = len(content)
        lines = []
        with open(outputfile + ".txt", "w") as fp_w:
            for i in range(linesnumber):
                values = content[i].split() if content[i].strip() != "" else None
                if values is None:
                    continue
                if values[0] == "ATOM" or values[0] == "HETATM":
                    value1 = values[2]  # atom_label
                    value2 = values[3]  # res_name
                    value3 = float(values[5])  # x
                    value4 = float(values[6])  # y
                    value5 = float(values[7])  # z
                    value6 = values[10]  # atom_note
                    value7 = int(values[4])
                    newline = "%7s%7s%5d%8.3f%8.3f%8.3f%7s" % (
                        value1,
                        value2,
                        value7,
                        value3,
                        value4,
                        value5,
                        value6,
                    )

                    lines.append(newline + "\n")
            fp_w.writelines(lines)
        data = pd.read_csv(
            outputfile + ".txt",
            sep="\s+",
            names=["Atom_label", "Residue", "Res_number", "x", "y", "z", "Note"],
        )
        return data

    def read_axis_from_lib(libfile):
        with open(libfile, "r") as f:
            lines = f.readlines()

        solution_1_2 = np.array(lines[0].split(), dtype=float)
        solution_1_3 = np.array(lines[2].split(), dtype=float)
        arr_1_2 = np.array(lines[1].split(), dtype=float).reshape(3, 3)
        arr_1_3 = np.array(lines[3].split(), dtype=float).reshape(3, 3)
        return solution_1_2, arr_1_2, solution_1_3, arr_1_3


class fetch:
    def reslist_resnum(node_local_pdb, linker_pdb, cut_lib_pdb):
        reslist, res_num = [], []
        file1 = read.pdb(node_local_pdb)
        reslist1 = file1["Residue"].unique()
        for i in reslist1:
            reslist.append(i)
        file2 = read.pdb(linker_pdb)
        reslist2 = file2["Residue"].unique()
        for i in reslist2:
            reslist.append(i)
        file3 = read.pdb(cut_lib_pdb)
        reslist3 = file3["Residue"].unique()
        for i in reslist3:
            reslist.append(i)

        for i in range(len(reslist1)):
            res_num.append(len(file1[file1["Residue"] == reslist1[i]]))
        for i in range(len(reslist2)):
            res_num.append(len(file2[file2["Residue"] == reslist2[i]]))
        for i in range(len(reslist3)):
            res_num.append(
                int(len(file3[file3["Residue"] == reslist3[i]]) / 12)
            )  # NOTE: number of repeated cut in lib file

        file_df = pd.concat([file1, file2, file3], ignore_index=True, join="outer")
        path = os.path.abspath("") + "/" + "Residues/"
        os.makedirs(path, exist_ok=True)

        for i in range(len(reslist)):
            with open(path + str(reslist[i]) + ".xyz", "w") as f:
                for index, row in file_df[file_df["Residue"] == reslist[i]].iterrows():
                    formatted_line = "%-5s%8.3f%8.3f%8.3f" % (
                        re.sub(r"\d", "", row["Atom_label"]),
                        row["x"],
                        row["y"],
                        row["z"],
                    )
                    f.write(formatted_line + "\n")

        return reslist, res_num


class output:
    def __init__(self, name, outpdb=True, outxyz=True, outgro=True):
        self.output = str(name)
        # self.begin_lines = useless_atom_number
        if outxyz:
            output.outxyz(self)
        if outgro:
            output.outgro(self)
        if outpdb:
            output.outpdb(self)

        """output"""

    def outgro(self):
        with open(self.output + ".txt", "r") as f:
            # Read the lines from the file
            lines = f.readlines()
            atoms_number = len(lines)
        newgro = []
        with open(self.output + ".gro", "w") as fp:
            newgro.append("generated by MOF_BUILD" + "\n" + str(atoms_number) + "\n")
            for i in range(atoms_number):
                values = lines[i].split()
                value_atom_number = int(i + 1)  # atom_number
                value_label = values[0]  # atom_label
                value_resname = values[1]  # residue_name
                value_resnumber = int(float(values[2]))  # residue number
                value_x = float(values[4]) / 10  # x
                value_y = float(values[5]) / 10  # y
                value_z = float(values[6]) / 10  # z
                formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
                    value_resnumber,
                    value_resname,
                    value_label,
                    value_atom_number,
                    value_x,
                    value_y,
                    value_z,
                )
                newgro.append(formatted_line + "\n")
            tail = "5 5 5 \n"
            newgro.append(tail)
            fp.writelines(newgro)

    def outxyz(self):
        with open(self.output + ".txt", "r") as f:
            lines = f.readlines()
            atoms_number = len(lines)
        newxyz = []
        with open(self.output + ".xyz", "w") as fp:
            newxyz.append(str(atoms_number) + "\n" + "generated by MOF_BUILD" + "\n")
            for i in range(atoms_number):
                values = lines[i].split()
                value_label = values[0]  # atom_label
                value_label = re.sub(r"\d", "", value_label)
                value_x = float(values[4])  # x
                value_y = float(values[5])  # y
                value_z = float(values[6])  # z
                formatted_line = "%-5s%8.3f%8.3f%8.3f" % (
                    value_label,
                    value_x,
                    value_y,
                    value_z,
                )
                newxyz.append(formatted_line + "\n")
            fp.writelines(newxyz)

    def outpdb(self):
        with open(self.output + ".txt", "r") as f:
            # Read the lines from the file
            lines = f.readlines()
            atoms_number = len(lines)

        newpdb = []
        with open(self.output + ".txt", "w") as fp:
            # Iterate over each line in the input file
            for i in range(atoms_number):
                # Split the line into individual values (assuming they are separated by spaces)
                values = lines[i].split()
                # Extract values based on their positions in the format string
                value1 = "ATOM"
                value2 = int(i + 1)
                value3 = values[0]  # label
                value4 = values[1]  # residue
                value5 = int(float(values[2]))  # residue number
                value6 = float(values[4])  # x
                value7 = float(values[5])  # y
                value8 = float(values[6])  # z
                value9 = "1.00"
                value10 = "0.00"
                value11 = values[3]  # note
                # Format the values using the specified format string
                formatted_line = "%-6s%5d%5s%4s%10d%8.3f%8.3f%8.3f%6s%6s%4s" % (
                    value1,
                    value2,
                    value3,
                    value4,
                    value5,
                    value6,
                    value7,
                    value8,
                    value9,
                    value10,
                    value11,
                )
                lines[i] = formatted_line + "\n"
            fp.writelines(lines)

        with open(self.output + ".pdb", "w") as fp:
            # Iterate over each line in the input file
            newpdb.append("generated by MOF_BUILD" + "\n")
            newpdb.append(lines[0])

            for i in range(1, len(lines)):
                lastline = lines[i - 1]
                thisline = lines[i]
                # Split the line into individual values (assuming they are separated by spaces)
                old_residue_number = lastline.split()[4]
                new_residue_number = thisline.split()[4]

                if old_residue_number != new_residue_number:
                    newline = "TER" + "\n" + thisline
                    newpdb.append(newline)
                else:
                    newline = thisline
                    newpdb.append(newline)
            fp.writelines(newpdb)


class save:
    def __init__(self) -> None:
        pass

    def all(df_node, df_linker, df_cut, outfilename, residues, res_count):
        df_all = pd.concat(
            [df_node, df_linker, df_cut], ignore_index=True, join="outer"
        )
        print("total atoms: " + str(len(df_all)))
        total_num = [1]
        new_df = pd.DataFrame()
        for i in range(len(residues)):
            df = df_all[df_all["Residue"] == residues[i]].reset_index(drop=True)
            df["Res_number"] = df.index // res_count[i] + total_num[-1]
            total_num.append(len(df) // res_count[i] + total_num[-1])
            print(str(residues[i]) + "   " + str(len(df) // res_count[i]))
            new_df = pd.concat([new_df, df], ignore_index=True, join="outer")
        new_df.to_csv(str(outfilename) + ".txt", sep="\t", header=None, index=False)
        return new_df


class pre_process:
    """pre-process"""

    def get_axis2(solution_1_2, arr_1_2, solution_1_3, arr_1_3):
        """
        this function is to calculate the operator from lib file axis1--> axis2 and apply this operator
        on dx to get the new_axis2 in new space
        """
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
        """this function is to generate a group of 3d SCATTER defined by user for further grouping points"""
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
        # find intersection of sets
        overlapped_elements = set1.intersection(set2)
        # convert back to array
        overlapped_array = np.array(list(overlapped_elements)).reshape(
            -1, array1.shape[-1]
        )
        return overlapped_array

    def find_solution(pAl1, pAl2, pAl1_1, pAl1_2, pAl1_3):
        Al1_Al2 = pAl2 - pAl1
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
        sin = float(np.sin(w))
        q_real = np.array([np.cos(w)])
        q_ijk = sin * axis
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
        # overlap = np.dot(v2_file,v1_file)*normalize_vector(v1_file)
        # n_v2_file=v2_file-overlap
        arr = (
            df_input.loc[:, ["x", "y", "z"]].to_numpy() - beginning_point
        )  # MOVE center (Co this case) to (0,0,0)
        q0 = rotate.calculate_q_rotation_with_vectors(v1_file, v1_frame)
        q1 = q0
        # if (q0 == quaternion.from_float_array([1,0,0,0]) and (np.dot(v1_file,v1_frame)<0)):
        #        q1 = quaternion.from_float_array([-1,0,0,0])
        # else:
        #       # q1 = quaternion.from_float_array(normalize_vector(quaternion.as_float_array(q0))) if abs(q0)>0 else q0
        #       #q1 = (quaternion.from_float_array([1, 0, 0, 0]) if (quaternion.as_float_array(q0)[0]> 1 )else q0)
        #        q1 = q0
        # print(q0,"q0",v1_file, v1_frame)
        # print(q1,"q1")
        q_V2 = quaternion.from_vector_part(v2_file)
        new_q_V2 = q1 * q_V2 * q1.inverse()
        new_V2_file = quaternion.as_vector_part(new_q_V2)
        angle = rotate.calculate_angle_rad(v1_frame, new_V2_file, v2_frame)
        q2 = rotate.calculate_q_rotation_with_axis_degree(v1_frame, angle)
        print(q2, "q2", angle)
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
        v1 = carte[0, :]  # dx
        v2 = tric[0, :]
        # axis = carte[2,:] #dz
        axis = normalize_vector(np.cross(v1, v2))
        theta = rotate.calculate_angle_rad(axis, v2, v1)
        r = rotate.calculate_q_rotation_with_axis_degree(axis, theta)
        # r = rotate.calculate_q_rotation_with_vectors(v1,v2)
        r_m = quaternion.as_rotation_matrix(r)
        new_tric_basis = np.round(np.dot(tric, r_m), 5)
        new_tric_points = np.dot(points, new_tric_basis)
        return new_tric_points


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


"""node linker cut"""


def get_node(Metal_file, tric_points_group, new_node_A, outputname):
    # rotate as group, translate as group
    Metal_file = read.pdb(Metal_file)
    Metal_count = 1
    zero_lines = new_node_A.shape[0]
    df_node = pd.DataFrame()
    for i in tric_points_group:
        new_positions = new_node_A + i
        df_left = pd.DataFrame(
            np.zeros((zero_lines, 4)),
            columns=["Atom_label", "Residue", "Res_number", "Note"],
        )
        df_left["Atom_label"] = Metal_file["Atom_label"]
        df_left["Residue"] = Metal_file["Residue"]
        df_left["Res_number"] = Metal_count
        df_left["Note"] = Metal_file["Note"]
        df_right = pd.DataFrame(new_positions, columns=["x", "y", "z"])
        df = pd.concat([df_left, df_right], axis=1, join="outer")
        df_node = pd.concat([df_node, df], ignore_index=True, join="outer")
        Metal_count += 1
    df_node.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_node


def calculate_linker(linker_file, linker_count, new_beginnings_array, new_linker):
    # translate by center points position, beginning point as CENTER OF PORPHYRIN like Co(body center of unit box)
    zero_lines = new_linker.shape[0]
    df_linker = pd.DataFrame()
    for i in new_beginnings_array:
        new_positions = new_linker + i
        df_left = pd.DataFrame(
            np.zeros((zero_lines, 4)),
            columns=["Atom_label", "Residue", "Res_number", "Note"],
        )
        df_left["Atom_label"] = linker_file["Atom_label"]
        df_left["Residue"] = linker_file["Residue"]
        df_left["Res_number"] = linker_count
        df_left["Note"] = linker_file["Note"]
        df_right = pd.DataFrame(new_positions, columns=["x", "y", "z"])
        df = pd.concat([df_left, df_right], axis=1, join="outer")
        df_linker = pd.concat([df_linker, df], ignore_index=True, join="outer")
        linker_count += 1
    return df_linker


def get_linker(
    t_basis,
    new_node_A,
    linker_pdb,
    O1_index,
    O2_index,
    O3_index,
    tric_points_c,
    outputname,
):  # rotate the linker to fit frame
    # ATTENTION:t_basis1 //short O-O,t_basis2//long O-O

    """
    rotate linker long O-O edge to the edge firstly then align the short O-O edge,
    accetable shift will be introduced by the O positions and angle calculation in this step
    """
    linker_file = read.pdb(linker_pdb)
    # O1 is the cross point
    O1, O2, O3 = (
        linker_file.loc[O1_index - 1, ["x", "y", "z"]].to_numpy(),
        linker_file.loc[O2_index - 1, ["x", "y", "z"]].to_numpy(),
        linker_file.loc[O3_index - 1, ["x", "y", "z"]].to_numpy(),
    )

    Z1, Z2, Z3, Z4, Z5, Z6 = (
        new_node_A[0, :],
        new_node_A[36, :],
        new_node_A[45, :],
        new_node_A[9, :],
        new_node_A[27, :],
        new_node_A[18, :],
    )

    t0 = np.cross(t_basis[2], t_basis[1])
    t1 = np.cross(t_basis[2], t_basis[0])
    t2 = np.cross(t_basis[1], t_basis[0])
    short_OO_basis_vector = np.linalg.norm(O2 - O1) * np.array(
        # [np.cross(t_basis[0],np.cross(t_basis[2],t_basis[1])), Z4 - Z3, Z1 - Z6, Z1 - Z2, Z1 - Z3, Z6 - Z2]
        [
            np.cross(t_basis[0], t0), #1
            np.cross(t_basis[1], t1),
            np.cross(t_basis[2], t2),
            Z2 - Z1,   #  y-x
            Z1 - Z3,#z-x
            Z6 - Z2, #z-y
        ]

        #     [Z3 - Z2, Z4 - Z3, Z1 - Z6, np.cross(t_basis[0],t_basis[1]),np.cross(t_basis[0],t_basis[2]),np.cross(t_basis[2],t_basis[1])]
    )
    long_OO_basis_vector = np.linalg.norm(O3 - O1) * np.array(
        [
            t_basis[0],
            t_basis[1],
            t_basis[2],
            t_basis[1] - t_basis[0],
            t_basis[2] - t_basis[0],
            t_basis[2] - t_basis[1],
        ]
    )
    # FIRST locate long edge THEN self rotate
    df_linker = pd.DataFrame()
    for i in range(6):
        if len(tric_points_c[i]) > 0:
            r1_vector_in_frame = normalize_vector(short_OO_basis_vector[i, :])
            r2_vector_in_frame = normalize_vector(long_OO_basis_vector[i, :])
            r1_vector_in_linker = normalize_vector(O2 - O1)  # short O-O
            r2_vector_in_linker = normalize_vector(O3 - O1)  # LONG O-O
            df_input = linker_file
            beginning_point = O1
            v1_file = r1_vector_in_linker
            v1_frame = r1_vector_in_frame
            v2_file = r2_vector_in_linker
            v2_frame = r2_vector_in_frame
            new_linker = rotate.rotate_twice_linker(
                df_input, beginning_point, v2_file, v2_frame, v1_file, v1_frame
            )
            # print("q1")
            # print(rotate.calculate_q_rotation_with_vectors(v2_file, v2_frame),v2_file,v2_frame)
            rotated_new_linker = new_linker - 0.5 * (
                new_linker[O2_index - 1] + new_linker[O3_index - 1]
            )  # make Co in beginning
            # rotated_new_linker = new_linker-center_of_new_linker # make Co in beginning
            linker_count = 1  # count number WILL be modified in df_all
            new_beginnings_array, new_linker = tric_points_c[i], rotated_new_linker
            df_linker1 = calculate_linker(
                linker_file, linker_count, new_beginnings_array, new_linker
            )
            if i > 0:
                df_linker = pd.concat([df_linker, df_linker1], axis=0, join="outer")
            else:
                df_linker = df_linker1
        else:
            continue
    df_linker.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_linker


def filt_boundary_cut_res(df_input, a, b):
    ab_axis = np.cross(a, b)
    r = rotate.calculate_q_rotation_with_vectors(ab_axis, np.array([1, 0, 0]))
    arr = df_input.loc[:, ["x", "y", "z"]].to_numpy()
    new_arr = rotate.get_rotated_array(arr, r)
    df = df_input[new_arr[:, 0] < -0.0]
    counts = df["Res_number"].value_counts()
    target_values_count = counts[
        counts == 3
    ].index  # FIXME:7 is for carboxylate based framewwork
    ab_filtered_df = df[df["Res_number"].isin(target_values_count)]
    return ab_filtered_df


def get_term(
    cut_lib_pdb,
    x_num,
    y_num,
    z_num,
    x_scalar,
    y_scalar,
    z_scalar,
    tric_basis,
    carte_points,
    tric_points,
    outputname,
):
    "use template cut file to add terminations to exposed metal node, we will treat the 6 faces separately"
    cut_term = read.pdb(cut_lib_pdb)
    group_A = carte_points
    x_max, y_max, z_max = x_num * x_scalar, y_num * y_scalar, z_num * z_scalar
    x_min, y_min, z_min = 0, 0, 0
    a, b, c = tric_basis[0], tric_basis[1], tric_basis[2]

    a_max_tric_points = tric_points[group_A[:, 0] == x_max]
    b_max_tric_points = tric_points[group_A[:, 1] == y_max]
    c_max_tric_points = tric_points[group_A[:, 2] == z_max]
    a_min_tric_points = tric_points[group_A[:, 0] == x_min]
    b_min_tric_points = tric_points[group_A[:, 1] == y_min]
    c_min_tric_points = tric_points[group_A[:, 2] == z_min]

    ab_filtered_min = filt_boundary_cut_res(cut_term, a, b)
    ac_filtered_min = filt_boundary_cut_res(cut_term, c, a)
    bc_filtered_min = filt_boundary_cut_res(cut_term, b, c)
    ab_filtered_max = filt_boundary_cut_res(cut_term, -a, b)
    ac_filtered_max = filt_boundary_cut_res(cut_term, -c, a)
    bc_filtered_max = filt_boundary_cut_res(cut_term, -b, c)

    extras = []
    for i in a_min_tric_points:
        extra_term = bc_filtered_min.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    for i in a_max_tric_points:
        extra_term = bc_filtered_max.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    for i in b_min_tric_points:
        extra_term = ac_filtered_min.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    for i in b_max_tric_points:
        extra_term = ac_filtered_max.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    for i in c_max_tric_points:
        extra_term = ab_filtered_max.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    for i in c_min_tric_points:
        extra_term = ab_filtered_min.loc[:, ["x", "y", "z"]].to_numpy() + i
        extras.append(extra_term)
    extras = np.concatenate(extras, axis=0)
    # cut_term['Residue']='CUT'

    left = []
    object_arr = cut_term.loc[
        :, ["Atom_label", "Residue", "Res_number", "Note"]
    ].to_numpy()
    for i in range(
        int(extras.shape[0] / 3)
    ):  # FIXME: 7 is for carboxylate based framewwork
        left.append(object_arr[0:3])  # FIXME:7 is for carboxylate based framewwork
    df_left = pd.DataFrame(
        (np.concatenate(left, axis=0)),
        columns=["Atom_label", "Residue", "Res_number", "Note"],
    )
    df_left["Res_number"] = (
        df_left.index // 3 + 1
    )  # FIXME:7 is for carboxylate based framewwork
    df_right = pd.DataFrame(extras, columns=["x", "y", "z"])
    df_cut = pd.concat([df_left, df_right], axis=1, join="outer")
    unique_cut_index = df_cut.loc[:, ["x", "y", "z"]].drop_duplicates().index
    df_cut = df_cut.loc[unique_cut_index]
    df_cut.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_cut
