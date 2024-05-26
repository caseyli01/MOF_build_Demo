def atoms2c(output,dx,dy,dz,x,y,z):
    with open(output+'.gro', 'r') as f:
        # Read the lines from the file
        lines = f.readlines()
        atoms_number = lines[1]

    newgro = []
    with open('new'+output+'.gro', 'w') as fp:
        newgro.append("generated by MOF_BUILD"+'\n'+str(atoms_number))
        # Iterate over each line in the input file
        for i in range (2,len(lines)-1):
            # Split the line into individual values (assuming they are separated by spaces)
            values = lines[i].split()
            value_resnumber = int((lines[i])[0:5])
            value_resname = lines[i][5:10]
            
            value_label = values[1] 
            value_atom_number = int(values[2]) 
            value_x = float(values[3])+dx #x      
            value_y = float(values[4])+dy#y
            value_z = float(values[5])+dz #z

            formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
                        value_resnumber, value_resname, value_label, value_atom_number, value_x, value_y, value_z
            )            
            newgro.append(formatted_line+'\n')
        
        box = str(x)+'   '+str(y)+'   '+str(z) + '\n'

        newgro.append(box)    

        fp.writelines(newgro)