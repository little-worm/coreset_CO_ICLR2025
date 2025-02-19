import os,sys
global my_WORK_DIR 
my_WORK_DIR = "./"
my_SEED = 1234
os.chdir(my_WORK_DIR)
sys.path.append(os.path.abspath("./catNips2024Code_0313"))
sys.path.append(os.path.abspath("my_dataCO"))
print(sys.path)