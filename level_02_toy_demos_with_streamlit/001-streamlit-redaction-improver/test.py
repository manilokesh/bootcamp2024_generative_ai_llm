
import os
import sys

print(os.path.realpath(__file__)) 
print(os.path.dirname(os.path.realpath(__file__))) 

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
print(parent_dir_path) 

sys.path.insert(0, parent_dir_path)


print(os.path.join(os.path.realpath(__file__), "../")) 
 

import os

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
 