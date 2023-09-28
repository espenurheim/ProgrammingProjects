from energyFunctions import *

#parameters
N1 = 8
M1 = 4

#Nodes with constraints
P = np.array([[1,1,0], [-1,1,0], [-1,-1,0], [1, -1, 0]], dtype = float).flatten()
#P = 0.2*np.array([[1,1,0], [-1,1,0], [-1,-1,0], [1, -1, 0]], dtype = float).flatten()

#Nodes without constraints. Initial value does not matter (convex function)
Xr = np.zeros((N1-M1,3), dtype = float)
Xr = np.array([[1,1,1], [-1,1,1], [-1,-1,1], [1, -1, 1]], dtype = float).flatten()
Xr = Xr.flatten()


#Every Node
X_init_1_dome = np.concatenate([P, Xr])

#Cables
C = np.zeros((N1,N1))
C[0,7] = 1
C[1,4] = 1
C[2,5] = 1
C[3,6] = 1
C[4,5] = 1
C[4,7] = 1
C[5,6] = 1
C[6,7] = 1

C = C + C.T

#Bars
B = np.zeros((N1,N1))
B[0,4] = 1
B[1,5] = 1
B[2,6] = 1
B[3,7] = 1

B = B + B.T

#Length of cables
L = np.zeros_like(C)
L[0,4] = 10
L[1,5] = 10 
L[2,6] = 10
L[3,7] = 10 
L[0,7] = 8
L[1,4] = 8
L[2,5] = 8
L[3,6] = 8
L[4,5] = 1
L[5,6] = 1
L[6,7] = 1
L[4,7] = 1

L = L + L.T

#External loads
el = np.zeros(N1)

# EXAMPLE 2 - pyramid showing non-convexity
M2 = 4
N2 = 5

P = np.array([[1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0]], dtype = float).flatten()
Xr_upper = np.array([0.5,0.5,1], dtype = float)
Xr_lower = np.array([0.5,0.5,-1], dtype = float)
X_dome_upper = np.concatenate([P, Xr_upper])
X_dome_lower = np.concatenate([P, Xr_lower])

C2 = np.zeros((N2,N2))

# Add bars in a pyramid-shape, all with lengths 2
B2 = np.zeros((N2,N2))
B2[0,4], B2[1,4], B2[2,4], B2[3,4] = 1, 1, 1, 1 
B2 = B2 + B2.T
L2 = B2.copy()*2

# No external loads
el2 = np.zeros(N2)