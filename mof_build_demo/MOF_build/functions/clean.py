import os
import glob
import sys
sys.path.append('..')
'''
clean  files
'''

class clean:
    def gro():
        files = glob.glob("*.gro")
        for f in files:
            os.remove(f)

    def xyz():
        files = glob.glob("*.xyz")
        for f in files:
            os.remove(f)

    def pdb():
        files = glob.glob("*.pdb")
        for f in files:
            os.remove(f)

    def csv():
        files = glob.glob("*.csv")
        for f in files:
            os.remove(f)

    def txt():
        files = glob.glob("*.txt")
        for f in files:
            os.remove(f)
