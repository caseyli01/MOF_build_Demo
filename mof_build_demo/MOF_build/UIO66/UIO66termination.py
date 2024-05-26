import pandas as pd
import numpy as np
from MOF_build.functions.read import read 
from MOF_build.functions.rotate import rotate

def filt_boundary_cut_res(df_input, a, b):
    ab_axis = np.cross(a, b)
    r = rotate.calculate_q_rotation_with_vectors(ab_axis, np.array([1, 0, 0]))
    arr = df_input.loc[:, ["x", "y", "z"]].to_numpy()
    new_arr = rotate.get_rotated_array(arr, r)
    df = df_input[new_arr[:, 0] < -0.0]
    counts = df["Res_number"].value_counts()
    target_values_count = counts[
        counts == 7
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
        int(extras.shape[0] / 7)
    ):  # FIXME: 7 is for carboxylate based framewwork
        left.append(object_arr[0:7])  # FIXME:7 is for carboxylate based framewwork
    df_left = pd.DataFrame(
        (np.concatenate(left, axis=0)),
        columns=["Atom_label", "Residue", "Res_number", "Note"],
    )
    df_left["Res_number"] = (
        df_left.index // 7 + 1
    )  # FIXME:7 is for carboxylate based framewwork
    df_right = pd.DataFrame(extras, columns=["x", "y", "z"])
    df_cut = pd.concat([df_left, df_right], axis=1, join="outer")
    unique_cut_index = df_cut.loc[:, ["x", "y", "z"]].drop_duplicates().index
    df_cut = df_cut.loc[unique_cut_index]
    df_cut.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_cut
