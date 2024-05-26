import pandas as pd
import numpy as np
from MOF_build.functions.rotate import rotate 
from MOF_build.functions.normv import normalize_vector
from MOF_build.functions.read import read

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
    x_scalar,
    y_scalar,
    z_scalar,
    dx,
    dy,
    dz,
    linker_pdb,
    O1_index,
    O2_index,
    O3_index,
    O4_index,
    points_c,
    outputname,
):  # rotate the linker to fit frame
    body_diag = (x_scalar * dx + y_scalar * dy + z_scalar * dz).ravel()

    center_of_Aunitbox_points = points_c - 0.5 * body_diag  # Co position

    """
    rotate porphyrin or other tetradentate linker to make it algin with dx and make Co position as parameter for 2nd rotate
    for further translation 
    """
    linker_file = read.pdb(linker_pdb)
    # O1 is the cross point
    O1, O2, O3, O4 = (
        linker_file.loc[O1_index - 1, ["x", "y", "z"]].to_numpy(),
        linker_file.loc[O2_index - 1, ["x", "y", "z"]].to_numpy(),
        linker_file.loc[O3_index - 1, ["x", "y", "z"]].to_numpy(),
        linker_file.loc[O4_index - 1, ["x", "y", "z"]].to_numpy()
    )
    Co = 0.5*(O1+O4) #center of O2 O3; beginning point
    r1_vector_in_frame = normalize_vector(dy)
    r2_vector_in_frame = normalize_vector(dz)
    r1_vector_in_linker = normalize_vector(O2 - O1)
    r2_vector_in_linker = normalize_vector(O3 - O1)

    df_input = linker_file
    beginning_point = Co
    v1_file = r1_vector_in_linker
    v1_frame = r1_vector_in_frame
    v2_file = r2_vector_in_linker
    v2_frame = r2_vector_in_frame

    new_linker = rotate.rotate_twice_linker(
        df_input, beginning_point, v1_file, v1_frame, v2_file, v2_frame
    )

    rotated_new_linker = new_linker  # make center in beginning
    linker_count = 1  # count number WILL be modified in df_all
    new_beginnings_array, new_linker = center_of_Aunitbox_points, rotated_new_linker
    df_linker = calculate_linker(
        linker_file, linker_count, new_beginnings_array, new_linker
    )
    df_linker.to_csv(str(outputname) + ".txt", header=None, sep="\t", index=False)
    return df_linker