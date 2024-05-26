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
