import re


class output:
    def __init__(self, name, outpdb=True, outxyz=True, outgro=True):
        self.output = str(name)
        # self.begin_lines = useless_atom_number
        if outxyz:
            output.outxyz(self)
        if outgro:
            output.outgro(self)
        if outpdb:
            output.outpdb(self)

        """output"""

    def outgro(self):
        with open(self.output + ".txt", "r") as f:
            # Read the lines from the file
            lines = f.readlines()
            atoms_number = len(lines)
        newgro = []
        with open(self.output + ".gro", "w") as fp:
            newgro.append("generated by MOF_BUILD" + "\n" + str(atoms_number) + "\n")
            for i in range(atoms_number):
                values = lines[i].split()
                value_atom_number = int(i + 1)  # atom_number
                value_label = values[0]  # atom_label
                value_resname = values[1]  # residue_name
                value_resnumber = int(float(values[2]))  # residue number
                value_x = float(values[4]) / 10  # x
                value_y = float(values[5]) / 10  # y
                value_z = float(values[6]) / 10  # z
                formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
                    value_resnumber,
                    value_resname,
                    value_label,
                    value_atom_number,
                    value_x,
                    value_y,
                    value_z,
                )
                newgro.append(formatted_line + "\n")
            tail = "5 5 5 \n"
            newgro.append(tail)
            fp.writelines(newgro)

    def outxyz(self):
        with open(self.output + ".txt", "r") as f:
            lines = f.readlines()
            atoms_number = len(lines)
        newxyz = []
        with open(self.output + ".xyz", "w") as fp:
            newxyz.append(str(atoms_number) + "\n" + "generated by MOF_BUILD" + "\n")
            for i in range(atoms_number):
                values = lines[i].split()
                value_label = values[0]  # atom_label
                value_label = re.sub(r"\d", "", value_label)
                value_x = float(values[4])  # x
                value_y = float(values[5])  # y
                value_z = float(values[6])  # z
                formatted_line = "%-5s%8.3f%8.3f%8.3f" % (
                    value_label,
                    value_x,
                    value_y,
                    value_z,
                )
                newxyz.append(formatted_line + "\n")
            fp.writelines(newxyz)

    def outpdb(self):
        with open(self.output + ".txt", "r") as f:
            # Read the lines from the file
            lines = f.readlines()
            atoms_number = len(lines)

        newpdb = []
        with open(self.output + ".txt", "w") as fp:
            # Iterate over each line in the input file
            for i in range(atoms_number):
                # Split the line into individual values (assuming they are separated by spaces)
                values = lines[i].split()
                # Extract values based on their positions in the format string
                value1 = "ATOM"
                value2 = int(i + 1)
                value3 = values[0]  # label
                value4 = values[1]  # residue
                value5 = int(float(values[2]))  # residue number
                value6 = float(values[4])  # x
                value7 = float(values[5])  # y
                value8 = float(values[6])  # z
                value9 = "1.00"
                value10 = "0.00"
                value11 = values[3]  # note
                # Format the values using the specified format string
                formatted_line = "%-6s%5d%5s%4s%10d%8.3f%8.3f%8.3f%6s%6s%4s" % (
                    value1,
                    value2,
                    value3,
                    value4,
                    value5,
                    value6,
                    value7,
                    value8,
                    value9,
                    value10,
                    value11,
                )
                lines[i] = formatted_line + "\n"
            fp.writelines(lines)

        with open(self.output + ".pdb", "w") as fp:
            # Iterate over each line in the input file
            newpdb.append("generated by MOF_BUILD" + "\n")
            newpdb.append(lines[0])

            for i in range(1, len(lines)):
                lastline = lines[i - 1]
                thisline = lines[i]
                # Split the line into individual values (assuming they are separated by spaces)
                old_residue_number = lastline.split()[4]
                new_residue_number = thisline.split()[4]

                if old_residue_number != new_residue_number:
                    newline = "TER" + "\n" + thisline
                    newpdb.append(newline)
                else:
                    newline = thisline
                    newpdb.append(newline)
            fp.writelines(newpdb)