from energyFunctions import *

# EXAMPLE 1
# parameters
N1 = 8
M = 0

#Nodes without constraints. Initial value does not matter (convex function)
X_init_1_free = np.array([[1,1,0], [-1,1,0], [-1,-1,0], [1, -1, 0],[-1,0,10], [0,-1,10], [1,0,10], [0, 1, 10]], dtype = float).flatten()

#Cables
C = np.zeros((N1,N1))
C[0,7], C[1,4], C[2,5], C[3,6], C[4,5], C[4,7], C[5,6], C[6,7], C[0,3], C[0,1], C[1,2], C[2,3] = np.ones(12)
C = C + C.T # Add transpose to make implementation easier

#Bars
B = np.zeros((N1,N1))
B[0,4], B[1,5], B[2,6], B[3,7] = np.ones(4)
B = B + B.T # Add transpose to make implementation easier

#Length of cables/bars
L = np.zeros_like(C)
L[0,4], L[1,5], L[2,6], L[3,7] = np.ones(4)*10
L[0,7], L[1,4], L[2,5], L[3,6] = np.ones(4)*8
L[4,5], L[5,6], L[6,7], L[4,7], L[0,3], L[0,1], L[1,2], L[2,3] = np.ones(8)*1
L = L + L.T # Add transpose to make implementation easier

#External loads
el = np.zeros(N1)

# EXAMPLE 2 - ANOTHER STACK
N2 = 12

#Nodes without constraints. Initial value does not matter (convex function)
X_init_2_free = np.array([[1,1,0], [-1,1,0], [-1,-1,0], [1, -1, 0],
                           [-0.7,0,9.5], [0,-0.7,9.5], [0.7,0,9.5], [0, 0.7, 9.5],
                           [-0.7,0,20], [0,-0.7,20], [0.7,0,20], [0, 0.7, 20]], dtype = float).flatten()

# Cables
C2 = np.zeros((N2,N2))
C2[0,7], C2[1,4], C2[2,5], C2[3,6], C2[4,5], C2[4,7], C2[5,6], C2[6,7], C2[0,3], C2[0,1], C2[1,2], C2[2,3], C2[4,11], C2[5,8], C2[6,9], C2[7,10], C2[8,9], C2[9,10], C2[10,11], C2[8,11] = np.ones(20)
C2 = C2 + C2.T # Add transpose to make implementation easier

# Bars
B2 = np.zeros((N2,N2))
B2[0,4], B2[1,5], B2[2,6], B2[3,7], B2[4,8], B2[5,9], B2[6,10], B2[7,11] = np.ones(8)
B2 = B2 + B2.T # Add transpose to make implementation easier

L2 = np.zeros_like(C2)
L2[0,4], L2[1,5], L2[2,6], L2[3,7], L2[4,8], L2[5,9], L2[6,10], L2[7,11] = np.ones(8)*10
L2[0,7], L2[1,4], L2[2,5], L2[3,6], L2[4,11], L2[5,8], L2[6,9], L2[7,10] = np.ones(8)*8
L2[4,5], L2[5,6], L2[6,7], L2[4,7], L2[0,3], L2[0,1], L2[1,2], L2[2,3], L2[8,9], L2[9,10], L2[10,11], L2[8,11] = np.ones(12)*1
L2 = L2 + L2.T # Add transpose to make implementation easier

el2 = np.zeros(N2)