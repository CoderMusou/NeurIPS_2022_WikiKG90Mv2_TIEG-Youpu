import os

import sys

os.system("cd dgl-ke-ogb-lsc/python;python3 -m pip install -e .")

os.system("pip install dgl==0.4.* -U")
# os.system("pip list")
try:
    import dgl
except:
    os.system("pip install dgl==0.4.* -U")
    os.system("pip install ogb")
