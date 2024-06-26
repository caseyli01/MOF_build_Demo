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
    t_basis, linker_pdb, O1_index, O2_index, O3_index, O4_index,tric_points_c,outputname
):  # rotate the linker to fit frame


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
        linker_file.loc[O4_index - 1, ["x", "y", "z"]].to_numpy(),
    )
    Co = 0.5*(O1+O4) #center of O2 O3; beginning point
    #short_OO_basis_vector = np.linalg.norm(O2 - O1) * np.array([t_basis[0], t_basis[0]])
    #short_OO_basis_vector = np.linalg.norm(O2 - O1) * np.array([t_basis[0], t_basis[0],t_basis[0]])
    #long_OO_basis_vector = np.linalg.norm(O3 - O1) * np.array([t_basis[1], t_basis[2],t_basis[1]-t_basis[2]])
    df_linker = pd.DataFrame()
    short_OO_basis_vector = np.linalg.norm(O2 - O1) * np.array([t_basis[0]])
    long_OO_basis_vector = np.linalg.norm(O3 - O1) * np.array([t_basis[1]])

    for i in range(6):  # 3 for 3 faces
        if len(tric_points_c[i]) > 0:
            r1_vector_in_frame = normalize_vector(short_OO_basis_vector[0, :])
            r2_vector_in_frame = normalize_vector(long_OO_basis_vector[0, :])
            r1_vector_in_linker = normalize_vector(O2 - O1)  # short O-O
            r2_vector_in_linker = normalize_vector(O3 - O1)  # LONG O-O
            
            df_input = linker_file
            beginning_point = Co
            
            v1_file = r1_vector_in_linker
            v1_frame = r1_vector_in_frame
            v2_file = r2_vector_in_linker
            v2_frame = r2_vector_in_frame

            new_linker = rotate.rotate_twice_linker(
                df_input, beginning_point,  v1_file, v1_frame,v2_file, v2_frame,
            )
            if i >0 :
                q_60i=rotate.calculate_q_rotation_with_axis_degree(t_basis[0],i*np.pi/3)
                new_linker = rotate.get_rotated_array(new_linker,q_60i)
           

            
                    # rotated_new_linker = new_linker-center_of_new_linker # make Co in beginning
            linker_count = 1  # count number WILL be modified in df_all
            new_beginnings_array, new_linker = tric_points_c[i], new_linker
            
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


if __name__ == "__main__":
    print('test')
    from functions.rotate import rotate
    from functions.output import output
    from PCN222frame import Frame
    from PCN222node import get_node
    from PCN222termination import get_term
    any_tdx, any_tdy, any_tdz = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 1, 2]))
    dx, dy, dz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    carte_basis = np.vstack([dx, dy, dz])
        # for uio66 fcb
    tric_basis = rotate.coordinate_transfer(any_tdx, any_tdy, any_tdz, carte_basis)
    x_num, y_num, z_num, x_scalar, y_scalar, z_scalar = 1,1,1,23,23,23
    libpath='D:\OneDrive - KTH\data\code\MOF_build_Demo\mof_build_demo\MOF_build\lib'
    node_axis_lib, node_local_pdb, cut_lib_pdb = libpath+'/PCN222_lib_axisZr.lib',libpath+'/PCN222_lib_nodeZr.lib',libpath+'/PCN222_lib_cut.lib'  
    frame = Frame(
            x_num, y_num, z_num, x_scalar, y_scalar, z_scalar, dx, dy, dz, tric_basis,node_axis_lib, node_local_pdb
        )
    carte_points, group_A, group_B, group_C, carte_points_c = frame.get_points()
    tric_points_c = []
    for i in range(3):  # 2 axis for 4 edge
            tric_points_c.append(np.dot(carte_points_c[i], tric_basis))
    O1_index, O2_index, O3_index, O4_index = 49,56,59,51
    linker_pdb = 'linkers4demo/M2.pdb'
    q_nodeA, new_node_B, new_node_A,  new_node_C = frame.node_learn_from_template()
    #print(group_A,group_B,group_C)

    df_node = get_node(
            node_local_pdb, group_A, group_B,group_C, tric_basis,new_node_A, new_node_B, new_node_C,'testnode'
        )

    df_linker = get_linker(
    tric_basis, linker_pdb, O1_index, O2_index, O3_index, O4_index,tric_points_c,'testlinker'
)#
    df_cut = get_term(
            cut_lib_pdb, x_num, y_num, z_num, x_scalar, y_scalar, z_scalar, tric_basis,q_nodeA,group_A, group_B, group_C,'outcut'
        )
    df_all = pd.concat(
            [df_node, df_linker,df_cut], ignore_index=True, join="outer"
        )
    df_all.to_csv( "testall.txt", sep="\t", header=None, index=False)
    output('testall', outgro=False,outpdb=False,outxyz=True)
    #print(tric_points_c)