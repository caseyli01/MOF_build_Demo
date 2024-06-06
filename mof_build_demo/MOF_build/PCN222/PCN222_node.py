import pandas as pd
import numpy as np
from MOF_build.functions.read import read 


def get_node(
    Metal_file, group_A, group_B,group_C, tric_basis, new_node_A, new_node_B, new_node_C,outputname
):
    # rotate as group, translate as group
    tric_group_A, tric_group_B, tric_group_C = (
        np.dot(group_A, tric_basis),
        np.dot(group_B, tric_basis),
        np.dot(group_C, tric_basis),
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
    for i in tric_group_C:
        new_positions = new_node_C + i
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

if __name__ == "__main__":
    from functions.rotate import rotate
    from functions.output import output
    from PCN222frame import Frame
    print('test')
    any_tdx, any_tdy, any_tdz = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 1]))
    dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    carte_basis = np.vstack([dx, dy, dz])
        # for uio66 fcb
    tric_basis = rotate.coordinate_transfer(any_tdx, any_tdy, any_tdz, carte_basis)
    x_num, y_num, z_num, x_scalar, y_scalar, z_scalar = 1,1,1,10,10,10
    libpath='D:\OneDrive - KTH\data\code\MOF_build_Demo\mof_build_demo\MOF_build\lib'
    node_axis_lib, node_local_pdb, cut_lib_pdb = libpath+'/MIL53_lib_axisAl.lib',libpath+'/MIL53_lib_nodeAl.lib',libpath+'/MIL53_lib_cut.lib' 
    frame = Frame(
            x_num, y_num, z_num, x_scalar, y_scalar, z_scalar, dx, dy, dz, tric_basis,node_axis_lib, node_local_pdb
        )
    carte_points, group_A, group_B, group_C, carte_points_c = frame.get_points()
    new_node_B, new_node_A,  new_node_C = frame.node_learn_from_template()
    print(group_A,group_B,group_C)

    df_node = get_node(
            node_local_pdb, group_A, group_B,group_C, tric_basis,new_node_A, new_node_B, new_node_C,'testnode'
        )
    output('testnode', outgro=False,outpdb=False,outxyz=True)
    print(df_node)