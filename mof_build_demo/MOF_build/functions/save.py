import pandas as pd


class save:
    """
    this class is to collect df_node df_linker df_cut and save them all as a new_df
    """

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
            if residues[i]=='ZRR':
                print(str(residues[i]) + "   " + str(6*(len(df) // res_count[i])))
            elif residues[i]=='OMM':
                print(str(residues[i]) + "   " + str(4*(len(df) // res_count[i])))
            elif residues[i]=='OHH':
                print(str(residues[i]) + "   " + str(4*(len(df) // res_count[i])))
            else:
                print(str(residues[i]) + "   " + str(len(df) // res_count[i]))
            new_df = pd.concat([new_df, df], ignore_index=True, join="outer")
        new_df.to_csv(str(outfilename) + ".txt", sep="\t", header=None, index=False)
        return new_df
    
    def noterm(df_node, df_linker,outfilename, residues, res_count):
        df_all = pd.concat(
            [df_node, df_linker], ignore_index=True, join="outer"
        )
        print("total atoms: " + str(len(df_all)))
        total_num = [1]
        new_df = pd.DataFrame()
        for i in range(len(residues)):
            df = df_all[df_all["Residue"] == residues[i]].reset_index(drop=True)
            df["Res_number"] = df.index // res_count[i] + total_num[-1]
            total_num.append(len(df) // res_count[i] + total_num[-1])
            if residues[i]=='ZRR':
                print(str(residues[i]) + "   " + str(6*(len(df) // res_count[i])))
            elif residues[i]=='OMM':
                print(str(residues[i]) + "   " + str(4*(len(df) // res_count[i])))
            elif residues[i]=='OHH':
                print(str(residues[i]) + "   " + str(4*(len(df) // res_count[i])))
            else:
                print(str(residues[i]) + "   " + str(len(df) // res_count[i]))
            new_df = pd.concat([new_df, df], ignore_index=True, join="outer")
        new_df.to_csv(str(outfilename) + ".txt", sep="\t", header=None, index=False)
        return new_df
