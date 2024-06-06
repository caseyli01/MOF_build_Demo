class replace_res():
    def __init__(self,Agro,Bgro,res_num,outputfilename):
        self.Agro = Agro
        self.Bgro = Bgro
        self.res_num = res_num
        self.outputname = outputfilename
        self.Agro_total_atoms_num = None
        self.Agro_extract_res_atoms_num = None
        self.Agro_head = None
        self.Agro_tail = None
        self.Agro_extracted_res = None
        self.Bgro_total_atoms_num = None
        self.Bgro_extract_res_atoms_num = None
        self.Bgro_head = None
        self.Bgro_tail = None
        self.Bgro_extracted_res = None

        
    def extractfileA(self):
        with open(self.Agro,'r') as f:
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

    def extractfileB(self):
        with open(self.Bgro,'r') as f:
            lines = f.readlines()
            total_atoms_num = int(lines[1].strip('\n'))
            self.Bgro_total_atoms_num =total_atoms_num
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
        self.Bgro_extract_res_atoms_num = residue_atoms_num
        self.Bgro_head = head_of_gro
        self.Bgro_tail = tail_of_gro
        self.Bgro_extracted_res = extract_res
    
    def extract_resBtoA(self):
        replace_res.extractfileA(self)
        replace_res.extractfileB(self)
        newlines = self.Agro_head+self.Bgro_extracted_res+self.Agro_tail
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
        print(str(self.Agro_extract_res_atoms_num)+" atoms in Agro file  \n"+str(self.Bgro_extract_res_atoms_num)+" atoms in Bgro file")
