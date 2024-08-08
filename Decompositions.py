import numpy as np
from scipy.linalg import svd
import torch
import tensorflow as tf

'''
This is a function that computes the singular value decomposition of
a given matrix.
Params: X (original matrix)
Returns: U (left singular vectors), sigma (matrix containing singular values),
VT (right singular vectors transposed)
'''
def SVD(X):
    XT = X.transpose()
    XTX = np.matmul(XT,X)
    
    eigenvals = np.linalg.eig(XTX)
    singular_vals = []
    for i in eigenvals[0]:
        if i >= 0:
            singular_vals.append(np.sqrt(i))
        else:
            singular_vals.append(np.sqrt(abs(i)))
            
    singular_vals.sort(reverse = True)

    count = 0
    for i in singular_vals:
        count += 1
    sigma = np.zeros((count, count))

    for i in range(count):
        sigma[i][i] = singular_vals[i]

    sigma_inv = np.linalg.inv(sigma)
    
    eigenvalues, eigenvectors = np.linalg.eigh(XTX)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    V = -1*eigenvectors[:, sorted_indices]
    VT = V.T
    
    U = np.matmul(X, np.matmul(V, sigma_inv))
    A = np.matmul(U, np.matmul(sigma, VT))

    checkU, checkS, checkVT = svd(X)
    if not np.allclose(U, checkU) or not np.allclose(sigma, np.diag(checkS)) or not np.allclose(VT, checkVT) or not np.allclose(A, X):
        print("There has been an issue computing the SVD.")
        return
    
    return U, sigma, VT

'''
This is a function that computes the Tucker Decomposition of an 
order-3 tensor using the Higher-Order Orthogonal Iteration 
Algorithm (HOOI) and the SVD program above.
Params: A (original tensor)
Returns: L, R, V (order matrices), B (core tensor)
'''
def Tucker(A):
    R = A[0]
    V = A[1]
    RT = R.T
    VT = V.T
    C = torch.mul(A, torch.mul(RT, VT))
    L, Ls, Lv = SVD(C[0].numpy()) #only need U from SVD
    LT = torch.tensor(L.T)
    D = torch.mul(A, torch.mul(LT, VT))
    R, Rs, Rv = SVD(D[1].numpy())
    E = torch.mul(A, torch.mul(LT, torch.tensor(R.T)))
    V, Vs, Vv = SVD(E[2].numpy())
    B = torch.mul(E, torch.tensor(V.T))
    return torch.tensor(L), torch.tensor(R), torch.tensor(V), B

print("How many rows do you want your tensor to have? ")
rows = input()
print("How many columns do you want your tensor to have? ")
cols = input()
print("How many units in the third dimension do you want your tensor to have? ")
dim3 = input()

X = np.zeros((int(rows), int(cols), int(dim3)))
for i in range(int(rows)):
    for j in range(int(cols)):
        for k in range(int(dim3)):
            print("enter the value for row " +str(i)+ " and column " +str(j)+ " and dimension " +str(k)+ ": ")
            X[i][j][k] = float(input())

print("The tensor you want to decompose is: ")      
print(X)
A = tf.convert_to_tensor(X)
print("Computing Tucker decomposition...")
L,R,V,core_tensor = Tucker(A)
print("The three order matrices are: ")
print(L)
print(R)
print(V)
print("The core tensor is: ")
print(core_tensor)
