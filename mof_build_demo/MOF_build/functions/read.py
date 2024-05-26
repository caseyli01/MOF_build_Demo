import numpy as np
import pandas as pd
import re
import sys
sys.path.append('.')
'''
read files
'''
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
