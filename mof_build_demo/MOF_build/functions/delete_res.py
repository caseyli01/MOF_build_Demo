class delete_res():
    def __init__(self,Agro,res_num,outputfilename):
        self.Agro = Agro
        self.res_num = res_num
        self.outputname = outputfilename
        self.Agro_total_atoms_num = None
        self.Agro_extract_res_atoms_num = None
        self.Agro_head = None
        self.Agro_tail = None
        self.Agro_extracted_res = None

        
    def extractfileA(self):
        with open(self.Agro+'.gro','r') as f:
            lines = f.readlines()
            total_atoms_num = int(lines[1].strip('\n'))
            self.Agro_total_atoms_num =total_atoms_num
        key = []
        for i in range(1,len(lines)-1):
            res_number = lines[i][0:5]
            res_number = int(res_number)
            if res_number == self.res_num:
                key.append(i)
        key.sort()
        residue_atoms_num = len(key)
        head_of_gro = lines[0:key[0]]
        tail_of_gro = lines[key[-1]+1:]
        extract_res = lines[key[0]:key[-1]+1]
        self.Agro_extract_res_atoms_num = residue_atoms_num
        self.Agro_head = head_of_gro
        self.Agro_tail = tail_of_gro
        self.Agro_extracted_res = extract_res
    

        
    
    def delete_resinA(self):
        delete_res.extractfileA(self)
        newlines = self.Agro_head+self.Agro_tail
        newlines[1]=str(len(newlines)-3)+'\n'
        for i in range(2,len(newlines)-1):
            atom_index = i-1
            line_head = newlines[i][0:15]
            line_left = newlines[i][20:]
            formatted_line = "%15s%5d%24s" % (
                    line_head,
                    atom_index,
                    line_left,
                )
            newlines[i]=formatted_line
        with open(str(self.outputname)+".gro",'w') as f:
            f.writelines(newlines)
        print(str(self.Agro_extract_res_atoms_num)+" atoms deleted in Agro file  \n")
    
    def reassign_res_num(self,wrong_gro_name,new_gro_name):
        with open(wrong_gro_name+'.gro','r') as f:
            lines = f.readlines()
        old_res_number_list =[]
        for i in range(2,len(lines)-1):
            old_res_number_list.append(int(lines[i][0:5]))
        unique_old_res_num = set(old_res_number_list)
        old_res_num_list = []
        for i in unique_old_res_num:
            old_res_num_list.append(i)
        print(str(len(unique_old_res_num))+"  residues in total")
        newlines = lines
        for i in range(2,len(lines)-1):
            old_res_number=int(lines[i][0:5])
            new_res_num=old_res_num_list.index(old_res_number)+1

            atom_index = i-1
            line_res_num = new_res_num
            line_between=lines[i][5:15]
            line_left = lines[i][20:]
            formatted_line = "%5d%10s%5d%24s" % (
                    line_res_num,
                    line_between,
                    atom_index,
                    line_left,
                )
            newlines[i]=formatted_line
            #print(formatted_line)
        
        with open(new_gro_name+'.gro','w') as f:
            f.writelines(newlines)
    