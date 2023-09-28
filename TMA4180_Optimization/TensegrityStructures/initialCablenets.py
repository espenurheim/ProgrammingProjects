from energyFunctions import *

# EXAMPLE 1
#parameters
N = 8
M1 = 4

#Nodes with constraints
P1 = np.array([[5,5,0], [-5,5,0], [-5,-5,0], [5, -5, 0]], dtype = float).flatten()

#Nodes without constraints. Initial value does not matter (convex function)
Xr = np.array([[1,1,0], [-1,1,0], [-1,-1,0], [1, -1, 0]], dtype = float).flatten()

#Every Node
X_init = np.concatenate([P1, Xr])

#Cables
C = np.zeros((N,N))
C[0,4] = 1
C[1,5] = 1
C[2,6] = 1
C[3,7] = 1
C[4,5] = 1
C[4,7] = 1
C[5,6] = 1
C[6,7] = 1
C = C + C.T

#Bars
B = np.zeros((N,N))

#Length of cables (here, all are 3)
L = 3*C.copy()

#External loads
el = np.zeros(N)
el[M1:] = np.ones(N-M1)*1/6

# EXAMPLE 2 (random)
M2 = 3
P = np.random.rand(M2,3)*10
Xr2 = np.random.rand(N-M2,3)*10
X_init_2 = np.concatenate([P.flatten(), Xr2.flatten()])
el = np.zeros(N)
el[M2:] = np.ones(N-M2)*1/6