import numpy as np
import ot
import networkx as nx

#def myMultiEmdGWD(argList = 10):    



def myMultiEmdGWD(arg):
    C_a, C_b = arg[0],arg[1]
    weight_a = np.ones(len(C_a)) / len(C_a)
    weight_b = np.ones(len(C_b)) / len(C_b)

    res = ot.gromov.gromov_wasserstein2(C_a,C_b,weight_a,weight_b)
    return res










# def test_myMultiEmdGWD():
#     costmatrix1 = nx.read_gpickle("/root/autodl-tmp/cat_nips24_mis_0728/my_dataCO/DIFCUSO_data/mis/er_test/ER_700_800_0.15_0.gpickle")
#     costmatrix2 = nx.read_gpickle("/root/autodl-tmp/cat_nips24_mis_0728/my_dataCO/DIFCUSO_data/mis/er_test/ER_700_800_0.15_2.gpickle")
#     arg = [costmatrix1,costmatrix2]
#     res = myMultiEmdGWD(arg)
#     print("res = ",res)

# test_myMultiEmdGWD()



