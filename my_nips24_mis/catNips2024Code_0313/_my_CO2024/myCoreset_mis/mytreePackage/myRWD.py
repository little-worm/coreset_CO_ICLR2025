import numpy as np
import ot
def myEmdRWD(location_a,weight_a,location_b,weight_b,maxIterTimes = 5):    
    # centralized location_a
    assert np.prod( np.abs(weight_a.dot(location_a) ) < 0.00001), "'location_a' must have 0 mean"
    assert np.prod( np.abs(weight_b.dot(location_b) ) < 0.00001), "'location_b' must have 0 mean"
    assert np.abs( sum(weight_a) - 1 ) < 0.00001, "'sum(weight_a)==1' must hold !!!"
    assert np.abs( sum(weight_b) - 1 ) < 0.00001, "'sum(weight_b)==1' must hold !!!"
    if sum(weight_a) < 1:
        index = np.argmax(weight_a)
        weight_a[index] += abs(1-sum(weight_a)) 
    else:
        index = np.argmax(weight_a)
        weight_a[index] -= abs(1-sum(weight_a))
    if sum(weight_b) < 1:
        index = np.argmax(weight_b)
        weight_b[index] += abs(1-sum(weight_b)) 
    else:
        index = np.argmax(weight_b)
        weight_b[index] -= abs(1-sum(weight_b))    
    ### 此处断言location_a,location_b是中心化过后的。
    for iterTime in range(maxIterTimes):
        #The following is for test---------------
        loss = ot.emd2(weight_a, weight_b, ot.dist(location_a,location_b))
        print("loss = ",loss)
        #----------------------------------------
        costMatrix = ot.dist(location_a,location_b)
        flowMartrix = ot.emd(weight_a, weight_b, costMatrix)
        matrixB = (location_a.T).dot(flowMartrix)
        matrixB = matrixB.dot(location_b)
        matrixU,matrixS,matrixVT = np.linalg.svd(matrixB)
        diagList = list([1 for i in range(len(matrixB)-1)])
        diagList.append(np.linalg.det(matrixU)*np.linalg.det(matrixVT))
        matrixR = matrixU.dot( np.diag(  diagList  ))
        matrixR = matrixR.dot(matrixVT)
        location_b = location_b.dot(matrixR.T)

    loss = ot.emd2(weight_a, weight_b, ot.dist(location_a,location_b))
    print('-'*50)
    return flowMartrix, location_b, loss






def test_myEmdRWD():
    n1 = 100
    n2 = 8
    d = 2
    location_a = np.random.rand(n1,d)*100
    location_b = np.random.rand(n2,d)*100
    weight_a = np.random.rand(n1); weight_a = weight_a / sum(weight_a)
    weight_b = np.random.rand(n2); weight_b = weight_b / sum(weight_b)
    location_a = location_a - weight_a.dot(location_a)
    location_b = location_b - weight_b.dot(location_b)

    myEmdRWD(location_a,weight_a,location_b,weight_b)


#test_myEmdRWD()







import numpy as np
import ot

def myMultiEmdRWD(argList = 10):    
    location_a,weight_a,location_b,weight_b,maxIterTimes = argList
    # centralized location_a
    assert np.prod( np.abs(weight_a.dot(location_a) ) < 0.00001), "'location_a' must have 0 mean"
    assert np.prod( np.abs(weight_b.dot(location_b) ) < 0.00001), "'location_b' must have 0 mean"
    assert np.abs( sum(weight_a) - 1 ) < 0.00001, "'sum(weight_a)==1' must hold !!!"
    assert np.abs( sum(weight_b) - 1 ) < 0.00001, "'sum(weight_b)==1' must hold !!!"
    if sum(weight_a) < 1:
        index = np.argmax(weight_a)
        weight_a[index] += abs(1-sum(weight_a)) 
    else:
        index = np.argmax(weight_a)
        weight_a[index] -= abs(1-sum(weight_a))
    if sum(weight_b) < 1:
        index = np.argmax(weight_b)
        weight_b[index] += abs(1-sum(weight_b)) 
    else:
        index = np.argmax(weight_b)
        weight_b[index] -= abs(1-sum(weight_b))    
    ### 此处断言location_a,location_b是中心化过后的。
    loss_pre = 10000000; loss_now = 10000000; iterTime = 0
    for iterTime in range(maxIterTimes):
        costMatrix = ot.dist(location_a,location_b)
        flowMartrix = ot.emd(weight_a, weight_b, costMatrix)
        loss_now = np.sum( np.array(costMatrix) * np.array(flowMartrix) )
    #    print("loss_now = ",loss_now)
        matrixB = (location_a.T).dot(flowMartrix)
        matrixB = matrixB.dot(location_b)
        matrixU,matrixS,matrixVT = np.linalg.svd(matrixB)
        diagList = list([1 for i in range(len(matrixB)-1)])
        diagList.append(np.linalg.det(matrixU)*np.linalg.det(matrixVT))
        matrixR = matrixU.dot( np.diag(  diagList  ))
        matrixR = matrixR.dot(matrixVT)
        location_b = location_b.dot(matrixR.T)
        if loss_pre - loss_now < 0.00001:
            break
        else:
            loss_pre = loss_now
    #print("iterTime = ",iterTime)
    return flowMartrix, location_b, loss_now






def test_myMultiEmdRWD():
    n1 = 100
    n2 = 8
    d = 2
    maxIterTimes = 5
    location_a = np.random.rand(n1,d)*100
    location_b = np.random.rand(n2,d)*100
    weight_a = np.random.rand(n1); weight_a = weight_a / sum(weight_a)
    weight_b = np.random.rand(n2); weight_b = weight_b / sum(weight_b)
    location_a = location_a - weight_a.dot(location_a)
    location_b = location_b - weight_b.dot(location_b)
    argList = [location_a,weight_a,location_b,weight_b,maxIterTimes,0.00001,None]
    myMultiEmdRWD(argList)


#   test_myMultiEmdRWD()



