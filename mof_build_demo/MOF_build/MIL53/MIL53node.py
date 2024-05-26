import pandas as pd
import numpy as np
from MOF_build.functions.read import read 


def get_node(
    Metal_file, group_A, group_B, tric_basis, new_node_A, new_node_B, outputname
):
    # rotate as group, translate as group
    tric_group_A, tric_group_B = (
        np.dot(group_A, tric_basis),
        np.dot(group_B, tric_basis),
    )
    Metal_file = read.pdb(Metal_file)
    Metal_count = 1
    zero_lines = new_node_A.shape[0]
    df_node = pd.DataFrame()
    for i in tric_group_A:
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
    for i in tric_group_B:
        new_positions = new_node_B + i
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

