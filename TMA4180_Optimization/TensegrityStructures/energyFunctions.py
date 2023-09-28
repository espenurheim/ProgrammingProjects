import numpy as np
import scipy.linalg as la

# CABLES AND EXTERNAL LOADS (CABLENETS) #####################################################################
#External load
def externalLoad(X,el):
    """
    - X:  Nodes
    - el: External load on the nodes (array)
    
    return Potential energy due to external load
    """
    X3 = X.reshape(int(len(X)/3),3)[:,2]
    return np.inner(el,X3)

def externalLoadGradient(X, el, M):
    """
    - X:  Nodes
    - el: External load on the nodes (array)
    - M:  Number of fixed nodes
    
    return Gradient of potential energy due to external load
    """
    grad = np.zeros_like(X).reshape(int(len(X)/3),3)
    grad[:,2] = el

    # RETURN GRADIENT OF FREE NODES:
    return grad[M:,:].flatten()

#Cable elasticity
def cableElast(X, C, L, k):
    """
    - X: Nodes
    - C: Cables
    - L: Lengths
    - k: Cable elasticity parameter

    return Potential energy due to cable elasticity
    """
    N, _ = C.shape
    X = X.reshape(N, 3)
    res = 0
    
    # Iterate over all nodes, add energy for each cable
    for i in range(N):
        for j in range(i+1, N):
            dist = la.norm(X[i] - X[j],2)
            if C[i, j] == 0 or dist <= L[i,j]:
                continue
            res += (k / (2*L[i, j]**2)) * (dist - L[i,j])**2
    return res

def cableElastGradient(X, C, L, k, M):
    """
    - X: Nodes
    - C: Cables
    - L: lengths
    - k: cable elasticity parameter
    - M: Number of fixed nodes

    return Gradient of potential energy due to cable elasticity (for the free nodes)
    """

    N, _ = C.shape
    X = X.reshape(N, 3)
    sol = np.zeros_like(X)
    
    for i in range(M,N):
        for j in range(N):
            dist = la.norm(X[i] - X[j],2)
            if C[i, j] == 0 or dist <= L[i,j]:
                continue
            sol[i] += (k / L[i, j]**2) * (1 - L[i,j] / dist) * (X[i] - X[j])
    

    # Return gradient for the free nodes only
    return sol[M:,:].flatten()

#Total energy in a cablenet
def energyCablenets(X, C, L, el, k):
    """
    - X:  Nodes
    - C:  Cables
    - L:  lengths
    - el: External load on the nodes
    - k:  cable elasticity parameter

    return Potential energy in a cablenet
    """

    Eel = externalLoad(X, el)
    Ece = cableElast(X, C, L, k)
    return Ece + Eel

def gradientCablenets(X, C, L, el, k, M):
    """
    - X:  Nodes
    - C:  Cables
    - L:  Lengths
    - el: External load on the nodes
    - k:  cable elasticity parameter
    - M:  Number of fixed nodes

    return Potential energy in a cablenet
    """

    EelGrad = externalLoadGradient(X, el, M)
    EceGrad = cableElastGradient(X, C, L, k, M)
    return EceGrad + EelGrad

# BARS (TENSEGRITY-DOMES) #############################################################################
#Bar gravity
def barGrav(X, B, L, rhog):
    """
    - X:   Nodes
    - B:   Bars
    - L:   Lengths
    - rhog: Bar density*g (gravity constant)
    
    return Gravitational potential energy of bars
    """
    N, _ = L.shape
    X = X.reshape(N, 3)
    res = 0
    for i in range(N):
        for j in range(i+1, N):
            if B[i,j] == 1:
                res += rhog*L[i,j]/2 * (X[i,2] + X[j,2])
                
    return res

def barGravGradient(X, B, L, rhog, M):
    """
    - X:    Nodes
    - B:    Bars
    - L:    Lengths
    - rhog: Bar density*g (gravity constant)
    - M:    Number of fixed nodes
    
    return Gradient of the gravitational potential energy of bars
    """  
    N, _ = B.shape
    X = X.reshape(N, 3)
    sol = np.zeros_like(X)

    for i in range(M,N):
        for j in range(N):
            if B[i,j] == 1:
                sol[i,2] += rhog*L[i,j]/2
    return sol[M:,:].flatten()

#Bar elasticity
def barElast(X, B, L, c):
    """
    - X: Nodes
    - B: Bars
    - L: lengths
    - c: bar elasticity parameter
    
    returns Potential energy due to bar elasticity
    """
    N, _ = B.shape
    X = X.reshape(N, 3)
    res = 0

    for i in range(N):
        for j in range(i+1, N):
            if B[i, j] == 1:
                X[:,2]
                dist = la.norm(X[i] - X[j],2)
                res += (c / (2*L[i, j]**2)) * (dist - L[i,j])**2
    return res

def barElastGradient(X, B, L, c, M):
    """
    - X: Nodes
    - B: Bars
    - L: lengths
    - c: Bar elasticity parameter
    - M: Number of fixed nodes
    
    returns Gradient of potential energy due to bar elasticity (for the free nodes)
    """
    N, _ = B.shape
    X = X.reshape(N, 3)
    sol = np.zeros_like(X)

    for i in range(M,N):
        for j in range(N):
            if B[i, j] == 1:
                dist = la.norm(X[i] - X[j],2)
                if dist != 0:
                    l = L[i,j]
                    sol[i] += (c / l**2) * (1 - l / dist) * (X[i] - X[j])
    
    return sol[M:,:].flatten()

#Total energy in a tensegrity dome
def energyTensegrityDomes(X, C, B, L, el, k, c, rhog):
    """
    - X:    Nodes
    - C:    Cables
    - B:    Bars
    - L:    lengths
    - el:   External load on the nodes
    - k:    cable elasticity parameter
    - c:    bar elasticity parameter
    - rhog: Bar density*g (gravity constant)

    return Potential energy in a cablenet
    """

    E_bar = barGrav(X, B, L, rhog) + barElast(X, B, L, c)
    E_cable = cableElast(X, C, L, k)
    E_ext = externalLoad(X, el)
    
    return E_bar + E_cable + E_ext

def gradientTensegrityDomes(X, C, B, L, el, k, c, rhog, M):
    """
    - X:    Nodes
    - C:    Cables
    - B:    Bars
    - L:    lengths
    - el:   External load on the nodes
    - k:    cable elasticity parameter
    - c:    bar elasticity parameter
    - rhog: Bar density*g (gravity constant)
    - M:    Number of fixed nodes

    return Potential energy in a cablenet
    """
    gradientBar = barGravGradient(X, B, L, rhog, M) + barElastGradient(X, B, L, c, M)
    gradientCable = cableElastGradient(X, C, L, k, M)
    gradientExt = externalLoadGradient(X, el, M)
    
    return gradientBar + gradientCable + gradientExt

# FREE-STANDING STRUCTURES ###########################################################################

# Penalty-energy
def quadraticPenaltyEnergy(X, mu1, mu2):
    """
    X:   Nodes
    mu1: Penalty parameter (inequality constraints)
    mu2: Penalty parameter (equality constraints)
    
    returns Penalty energy of the free-standing structure
    """
    penalty = np.zeros_like(X)
    penalty[2::3] = mu1/2 * np.maximum(np.zeros_like(X[2::3]), -X[2::3])**2
    penalty[0:2] = mu2/2 * (X[0:2]-np.ones(2))**2
    return np.sum(penalty)

# Penalty-energy gradient
def quadraticPenaltyEnergyGradient(X, mu1, mu2):
    """
    X:   Nodes
    mu1: Penalty parameter (inequality constraints)
    mu2: Penalty parameter (equality constraints)
    
    returns Gradient of the penalty energy of the free-standing structure
    """
    
    penalty_gradient = np.zeros_like(X)
    penalty_gradient[2::3] = -mu1 * np.maximum(np.zeros_like(X[2::3]), -X[2::3])
    penalty_gradient[0:2] = mu2 * (X[0:2]-np.ones(2))
    
    return penalty_gradient

# Total energy in a free-standing structure:
def energyFreeStanding(X, C, B, L, el, k, c, rhog, M, mu1, mu2):
    """
    - X:  Nodes
    - C:  Cables
    - B:  Bars
    - L:  lengths
    - el: External load on the nodes
    - k:  cable elasticity parameter
    - c:  bar elasticity parameter
    - rhog: Bar density*g (gravity constant)
    - M:  Number of fixed nodes
    - mu1: Penalty parameter (inequality constraints)
    - mu2: Penalty parameter (equality constraints)

    return Potential energy in a cablenet
    """
    E_bar = barGrav(X, B, L, rhog) + barElast(X, B, L, c)
    E_cable = cableElast(X, C, L, k)
    E_ext = externalLoad(X, el)
    E_penalty = quadraticPenaltyEnergy(X, mu1, mu2)
    
    return E_bar + E_cable + E_ext + E_penalty

# Total energy gradient in a free-standing structure:
def gradientFreeStanding(X, C, B, L, el, k, c, rhog, M, mu1, mu2):
    """
    - X:  Nodes
    - C:  Cables
    - B:  Bars
    - L:  lengths
    - el: External load on the nodes
    - k:  cable elasticity parameter
    - c:  bar elasticity parameter
    - rhog: Bar density*g (gravity constant)
    - M:  Number of fixed nodes
    - mu1: Penalty parameter (inequality constraints)
    - mu2: Penalty parameter (equality constraints)

    return Potential energy in a cablenet
    """
    gradientBar = barGravGradient(X, B, L, rhog, M) + barElastGradient(X, B, L, c, M)
    gradientCable = cableElastGradient(X, C, L, k, M)
    gradientExt = externalLoadGradient(X, el, M)
    gradientPenalty = quadraticPenaltyEnergyGradient(X, mu1, mu2)
    
    return gradientBar + gradientCable + gradientExt + gradientPenalty