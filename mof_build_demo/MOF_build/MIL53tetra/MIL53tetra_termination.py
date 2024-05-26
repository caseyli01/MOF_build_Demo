import pandas as pd
import numpy as np
from MOF_build.functions.read import read 
def get_term(
    cut_lib_pdb,
    x_num,
    y_num,
    z_num,
    x_scalar,
    y_scalar,
    z_scalar,
    group_A,
    group_B,
    outputname,
):
    cut_term = read.pdb(cut_lib_pdb)
    x_max, y_max, z_max = x_num * x_scalar, y_num * y_scalar, z_num * z_scalar
    x_min, y_min, z_min = 0, 0, 0
    x_max_series_groupA, x_max_series_groupB = (
        group_A[group_A[:, 0] == x_max],
        group_B[group_B[:, 0] == x_max],
    )
    # y_max_series_groupA,y_max_series_groupB = group_A[group_A[:, 1] == y_max],group_B[group_B[:, 1] == y_max]
    # z_max_series_groupA,z_max_series_groupB = group_A[group_A[:, 2] == z_max],group_B[group_B[:, 2] == z_max]

    x_min_series_groupA, x_min_series_groupB = (
        group_A[group_A[:, 0] == x_min],
        group_B[group_B[:, 0] == x_min],
    )
    # y_min_series_groupA,y_min_series_groupB = group_A[group_A[:, 1] == y_min],group_B[group_B[:, 1] == y_min]
    # z_min_series_groupA,z_min_series_groupB = group_A[group_A[:, 2] == z_min],group_B[group_B[:, 2] == z_min]
    # x_max_series_groupA,x_max_series_groupB = group_A[group_A[:, 0] == x_max],group_B[group_B[:, 0] == x_max]
    y_max_series_groupA, y_max_series_groupB = (
        group_A[(group_A[:, 1] == y_max) & (group_A[:, 0] != x_max)],
        group_B[(group_B[:, 1] == y_max) & (group_B[:, 0] != x_min)],
    )
    z_max_series_groupA, z_max_series_groupB = (
        group_A[(group_A[:, 2] == z_max) & (group_A[:, 0] != x_max)],
        group_B[(group_B[:, 2] == z_max) & (group_B[:, 0] != x_max)],
    )

    # x_min_series_groupA,x_min_series_groupB = group_A[group_A[:, 0] == x_min],group_B[group_B[:, 0] == x_min]
    y_min_series_groupA, y_min_series_groupB = (
        group_A[(group_A[:, 1] == y_min) & (group_A[:, 0] != x_min)],
        group_B[(group_B[:, 1] == y_min) & (group_B[:, 0] != x_max)],
    )
    z_min_series_groupA, z_min_series_groupB = (
        group_A[(group_A[:, 2] == z_min) & (group_A[:, 0] != x_min)],
        group_B[(group_B[:, 2] == z_min) & (group_B[:, 0] != x_min)],
    )
    B_A_rotation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    cut_term_xyz_group_A = cut_term.loc[:, ["x", "y", "z"]].to_numpy()
    cut_term_xyz_group_B = np.dot(cut_term_xyz_group_A, B_A_rotation_matrix)

    extras = []
    for i in x_min_series_groupA:
        temp = np.dot(
            cut_term_xyz_group_A[cut_term_xyz_group_A[:, 0] < 0],
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        )
        extra_term = temp - np.array([0, 0, temp[1, 2]]) + i
        extras.append(extra_term)
    for i in x_max_series_groupA:
        temp = np.dot(
            cut_term_xyz_group_A[cut_term_xyz_group_A[:, 0] > 0],
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        )
        extra_term = temp - np.array([0, 0, temp[1, 2]]) + i
        extras.append(extra_term)
    for i in y_min_series_groupA:
        # if i[0]>x_min:
        extra_term = cut_term_xyz_group_A[cut_term_xyz_group_A[:, 1] < 0] + i
        extras.append(extra_term)
    for i in y_max_series_groupA:
        # if i[0]<x_max:
        extra_term = cut_term_xyz_group_A[cut_term_xyz_group_A[:, 1] > 0] + i
        extras.append(extra_term)
    for i in z_min_series_groupA:
        # if i[0]>x_min:
        extra_term = cut_term_xyz_group_A[cut_term_xyz_group_A[:, 2] < 0] + i
        extras.append(extra_term)
    for i in z_max_series_groupA:
        # if i[0]<x_max:
        extra_term = cut_term_xyz_group_A[cut_term_xyz_group_A[:, 2] > 0] + i
        extras.append(extra_term)

    for i in x_min_series_groupB:
        temp = np.dot(
            cut_term_xyz_group_B[cut_term_xyz_group_B[:, 0] < 0],
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        )
        extra_term = temp - np.array([0, 0, temp[1, 2]]) + i
        extras.append(extra_term)
    for i in x_max_series_groupB:
        temp = np.dot(
            cut_term_xyz_group_B[cut_term_xyz_group_B[:, 0] > 0],
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        )
        extra_term = temp - np.array([0, 0, temp[1, 2]]) + i
        extras.append(extra_term)
    for i in y_min_series_groupB:
        # if i[0]<x_max:
        extra_term = cut_term_xyz_group_B[cut_term_xyz_group_B[:, 1] < 0] + i
        extras.append(extra_term)
    for i in y_max_series_groupB:
        # if i[0]>x_min:
        extra_term = cut_term_xyz_group_B[cut_term_xyz_group_B[:, 1] > 0] + i
        extras.append(extra_term)
    for i in z_min_series_groupB:
        # if i[0]>x_min:
        extra_term = cut_term_xyz_group_B[cut_term_xyz_group_B[:, 2] < 0] + i
        extras.append(extra_term)
    for i in z_max_series_groupB:
        # if i[0]<x_max:
        extra_term = cut_term_xyz_group_B[cut_term_xyz_group_B[:, 2] > 0] + i
        extras.append(extra_term)

    extras = np.concatenate(extras, axis=0)
    cut_term["Residue"] = "CUT"

    left = []
    object_arr = cut_term.loc[
        :, ["Atom_label", "Residue", "Res_number", "Note"]
    ].to_numpy()
    for i in range(int(extras.shape[0] / 7)):
        left.append(object_arr[0:7])
    df_left = pd.DataFrame(
        (np.concatenate(left, axis=0)),
        columns=["Atom_label", "Residue", "Res_number", "Note"],
    )
    df_left["Res_number"] = df_left.index // 7 + 1

    df_right = pd.DataFrame(extras, columns=["x", "y", "z"])
    df_cut = pd.concat([df_left, df_right], axis=1, join="outer")
    unique_cut_index = df_cut.loc[:, ["x", "y", "z"]].drop_duplicates().index
    df_cut = df_cut.loc[unique_cut_index]
    df_cut.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_cut
