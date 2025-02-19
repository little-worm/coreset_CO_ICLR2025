import glob
import networkx as nx
import numpy as np
import os,ot
from multiprocessing import Pool
 

def create_costmatrix(pickleFilename):
    graph = nx.read_gpickle(pickleFilename)
    costmatrix = np.zeros((len(graph),len(graph)))
    for i in range(len(graph)):
        for j in range(len(graph)):
            costmatrix[i,j] = nx.shortest_path_length(graph,i,j) 
    save_name = os.path.basename(pickleFilename)[0:-7] + "costmatrix"      
    np.save(gpickle_costmatrixPath + "/" + save_name, costmatrix)  
    print("done!!!====",pickleFilename)
    
    
    
    
    
myPoolNum = 20
gpickle_folderPath = "/root/autodl-tmp/cat_nips24_mis/catNips2024Code_0313/DIFUSCO_main/data/mis-benchmark-framework/my_dataCO/DIFCUSO_data/mis/"
gpickle_costmatrixPath = gpickle_folderPath + "costmatrix_file"
if not os.path.exists(gpickle_costmatrixPath):
    os.makedirs(gpickle_costmatrixPath)
    print(f"The directory {gpickle_costmatrixPath } has been created.")
else:
    print(f"The directory {gpickle_costmatrixPath } already exists.")

global_pickleFilename_list = glob.glob(gpickle_folderPath + "*gpickle")

with Pool(myPoolNum) as pool:
    pool.map(create_costmatrix,global_pickleFilename_list) 



            
            
            
            
            










            