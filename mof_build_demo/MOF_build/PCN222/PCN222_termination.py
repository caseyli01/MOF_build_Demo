import pandas as pd
import numpy as np
from MOF_build.functions.rotate import rotate
from MOF_build.functions.read import read 

def filt_boundary_cut_res_above(df_input, a, b):
    ab_axis = np.cross(a, b)
    r = rotate.calculate_q_rotation_with_vectors(ab_axis, np.array([1, 0, 0]))
    arr = df_input.loc[:, ["x", "y", "z"]].to_numpy()
    new_arr = rotate.get_rotated_array(arr, r)
    df = df_input[new_arr[:, 0] < 0]
    counts = df["Res_number"].value_counts()
    target_values_count = counts[
        counts ==7
    ].index  # FIXME:7 is for carboxylate based framewwork
    ab_filtered_df = df[df["Res_number"].isin(target_values_count)]
    #this will lead to overlap but can find unique terms later
    return ab_filtered_df

def filt_boundary_cut_res_below(df_input, a, b):
    ab_axis = np.cross(a, b)
    r = rotate.calculate_q_rotation_with_vectors(ab_axis, np.array([1, 0, 0]))
    arr = df_input.loc[:, ["x", "y", "z"]].to_numpy()
    new_arr = rotate.get_rotated_array(arr, r)
    df = df_input[new_arr[:, 0] > 0]
    counts = df["Res_number"].value_counts()
    target_values_count = counts[
        counts == 7
    ].index  # FIXME:7 is for carboxylate based framewwork
    ab_filtered_df = df[df["Res_number"].isin(target_values_count)]
    #this will lead to overlap but can find unique terms later
    return ab_filtered_df

def filter_abc_terminations(cut_term,a,b,c,carte_points,tric_points,x_max,y_max,z_max,x_min,y_min,z_min):
    a_max_tric_points = tric_points[carte_points[:, 0] == x_max]
    b_max_tric_points = tric_points[carte_points[:, 1] == y_max]
    c_max_tric_points = tric_points[carte_points[:, 2] == z_max]
    a_min_tric_points = tric_points[carte_points[:, 0] == x_min]
    b_min_tric_points = tric_points[carte_points[:, 1] == y_min]
    c_min_tric_points = tric_points[carte_points[:, 2] == z_min]
    ab_filtered_min = filt_boundary_cut_res_above(cut_term, b, a)
    ac_filtered_min = filt_boundary_cut_res_above(cut_term, c, a)
    bc_filtered_min = filt_boundary_cut_res_above(cut_term, b, c)
    ab_filtered_max = filt_boundary_cut_res_below(cut_term, b, a)
    ac_filtered_max = filt_boundary_cut_res_below(cut_term, c, a)
    bc_filtered_max = filt_boundary_cut_res_below(cut_term, b, c)
    #print(a_max_tric_points.shape,b_max_tric_points.shape,c_max_tric_points.shape)

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
    return extras
def anti_filter_abc_terminations(cut_term,a,b,c,carte_points,tric_points,x_max,y_max,z_max,x_min,y_min,z_min):
    a_max_tric_points = tric_points[carte_points[:, 0] == x_max]
    b_max_tric_points = tric_points[carte_points[:, 1] == y_max]
    c_max_tric_points = tric_points[carte_points[:, 2] == z_max]
    a_min_tric_points = tric_points[carte_points[:, 0] == x_min]
    b_min_tric_points = tric_points[carte_points[:, 1] == y_min]
    c_min_tric_points = tric_points[carte_points[:, 2] == z_min]
    ab_filtered_min = filt_boundary_cut_res_below(cut_term, b, a)
    ac_filtered_min = filt_boundary_cut_res_above(cut_term, c, a)
    bc_filtered_min = filt_boundary_cut_res_above(cut_term, b, c)
    ab_filtered_max = filt_boundary_cut_res_above(cut_term, b, a)
    ac_filtered_max = filt_boundary_cut_res_below(cut_term, c, a)
    bc_filtered_max = filt_boundary_cut_res_below(cut_term, b, c)
    #print('ab',ab_filtered_min,ab_filtered_max)
    #print('ac',ac_filtered_min,ac_filtered_max)
    #print('bc',bc_filtered_min,bc_filtered_max)

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
    return extras


def merge_df_left_right(template_df,xyz_array):
    df_left = pd.DataFrame(
                np.zeros((template_df.shape[0], 4)),
                columns=["Atom_label", "Residue", "Res_number", "Note"],
            )
    df_left["Atom_label"] = template_df["Atom_label"]
    df_left["Residue"] = template_df["Residue"]
    df_left["Res_number"] = template_df["Res_number"]
    df_left["Note"] = template_df["Note"]
    df_right = pd.DataFrame(xyz_array, columns=["x", "y", "z"])
    df = pd.concat([df_left, df_right], axis=1, join="outer")
    return df


def get_term(
    cut_lib_pdb,
    q_rotate_local_node,
    x_num,
    y_num,
    z_num,
    x_scalar,
    y_scalar,
    z_scalar,
    tric_basis,
    carte_groupA,
    carte_groupB,
    carte_groupC,
    outputname,
):
    "use template cut file to add terminations to exposed metal node, we will treat the 6 faces separately"
    cut_term = read.pdb(cut_lib_pdb)
    
    x_max, y_max, z_max = x_num *x_scalar, y_num * 2* y_scalar, z_num *2* z_scalar
    x_min, y_min, z_min = 0, 0, 0
    a, b, c = tric_basis[0], tric_basis[1], tric_basis[2]

    
    
    #carte_groupA--tric_groupA
    tric_groupA = np.dot(carte_groupA, tric_basis)
    tric_groupB = np.dot(carte_groupB, tric_basis)
    tric_groupC = np.dot(carte_groupC, tric_basis)
    #rotated cut_term
    angle = -1*np.pi/3
    q_rotate = rotate.calculate_q_rotation_with_axis_degree(tric_basis[0], angle)
    cut_term_array = rotate.get_rotated_array(cut_term.loc[:,["x", "y", "z"]].to_numpy(),q_rotate_local_node)
    cut_term.loc[:,["x", "y", "z"]]=cut_term_array

   
    cut_termB_array = cut_term_array
    cut_termA_array = rotate.get_rotated_array(cut_termB_array, q_rotate)
    cut_termC_array = rotate.get_rotated_array(cut_termA_array, q_rotate)
    
    cut_termA = merge_df_left_right(cut_term,cut_termA_array)
    cut_termB = merge_df_left_right(cut_term,cut_termB_array)
    cut_termC = merge_df_left_right(cut_term,cut_termC_array)


    extrasA = filter_abc_terminations(cut_termA,a,b,c,carte_groupA,tric_groupA,x_max,y_max,z_max,x_min,y_min,z_min)
    extrasB = filter_abc_terminations(cut_termB,a,b,c,carte_groupB,tric_groupB,x_max,y_max,z_max,x_min,y_min,z_min)
    #extrasC = filter_abc_terminations(cut_termC,a,b,c,carte_groupC,tric_groupC,x_max,y_max,z_max,x_min,y_min,z_min)
    extrasC=anti_filter_abc_terminations(cut_termC,a,b,c,carte_groupC,tric_groupC,x_max,y_max,z_max,x_min,y_min,z_min)
    # cut_term['Residue']='CUT'
    extras = np.concatenate((extrasA,extrasB,extrasC), axis=0)

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

