import numpy as np
import matplotlib.pyplot as plt

# Wolfe linesearch method
def strongWolfe(f, grad_f, X, p, M,
                initial_value,
                initial_descent,
                alpha_0 = 1.0,
                c1 = 1e-2,
                c2 = 0.9,
                rho = 2.0,
                max_ext_it = 100,  # Maximum extrapolation iterations
                max_int_it = 30): # Maximum interpolation iterations
    '''
    Implementation of a bisection based bracketing method
    for the strong Wolfe conditions.
    '''
    Xa = X # All X, including fixed nodes
    X = X[(3*M):] # Only free nodes
    
    # Cnitialize bounds of the bracketing interval
    alphaR = alpha_0
    alphaL = 0.0
    
    # Calculate the function value at the next point
    next_X = X+alphaR*p
    next_Xa = Xa
    next_Xa[(3*M):] = next_X
    next_value = f(next_Xa)
    
    # Check Armijo and Wolfe conditions
    next_grad = grad_f(next_Xa)
    Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
    descentR = np.inner(p,next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)

    # Increase upper bound as long as Armijo and curvatureHigh hold, but curvatureLow fails.
    it = 0
    while (it < max_ext_it and (Armijo and (not curvatureLow))):
        it += 1
        # alphaR is a new lower bound, increase upper bound by a factor rho
        alphaL = alphaR
        alphaR *= rho
        
        # Update function value and gradient
        next_X = X+alphaR*p
        next_Xa = Xa
        next_Xa[(3*M):] = next_X
        next_value = f(next_Xa)
        next_grad = grad_f(next_Xa)
        
        # Update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    
    alpha = alphaR
    
    it = 0
    # Use bisection to find a step length alpha that satisfies all conditions.
    while (it < max_int_it and (not (Armijo and curvatureLow and curvatureHigh))):
        it += 1
        if (Armijo and (not curvatureLow)):
            # alpha was too small
            alphaL = alpha
        else:
            # alpha was too large
            alphaR = alpha
            
        # choose a new step length as the mean of the new bounds
        alpha = (alphaL+alphaR)/2
        
        # update function value and gradient
        next_X = X+alphaR*p
        next_Xa = Xa
        next_Xa[(3*M):] = next_X
        next_value = f(next_Xa)
        next_grad = grad_f(next_Xa)
        
        # update the Armijo and Wolfe conditions
        Armijo = (next_value <= initial_value+c1*alphaR*initial_descent)
        descentR = np.inner(p,next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
        
    return next_Xa, next_value, next_grad


# FUNCTION TO PRINT RESULTS
def print_result(X, N, M):
    print("\nNumerical solution for the free nodes:")
    for i in range(M, N):
        print(f"x{i+1} = {X[i]}")
    print("")

# FUNCTION TO PLOT RESULTS
def plot_result(X_init, X, C, B, N, M, title = "Numerical solution"):
    '''
    Make 3D-plot of a tensegrity structure.
    '''
    #fig = plt.subplots
    fig = plt.figure(figsize = (16,7))
    fig.suptitle(title, fontsize=20)
    
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.set_title("Initial structure", fontsize=15)
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$x_3$")

    # Plot nodes for first structure
    for i in range(N):
        if i < M:
            ax1.scatter(X_init[i,0], X_init[i,1], X_init[i,2], s = 100, c = 'k')
        else:
            ax1.scatter(X_init[i,0], X_init[i,1], X_init[i,2], s = 100, c = 'r')
    
    # Plot cables and bars for first structure
    for i in range(N):
        for j in range(i, N):
            xs = [X_init[i,0], X_init[j,0]]
            ys = [X_init[i,1], X_init[j,1]]
            zs = [X_init[i,2], X_init[j,2]]
            
            if C[i,j] == 1:
                ax1.plot(xs, ys, zs, '--', c = 'k')
                
            if B[i,j] == 1:
                ax1.plot(xs, ys, zs, '-', c = 'k')
                
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    ax2.set_title("Numerical solution", fontsize=15)
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$x_3$")

    # Plot nodes for second structure
    for i in range(N):
        if i < M:
            ax2.scatter(X[i,0], X[i,1], X[i,2], s = 100, c = 'k')
        else: 
            ax2.scatter(X[i,0], X[i,1], X[i,2], s = 100, c = 'r')
    
    # Plot cables and bars for second structure
    for i in range(N):
        for j in range(i, N):
            xs = [X[i,0], X[j,0]]
            ys = [X[i,1], X[j,1]]
            zs = [X[i,2], X[j,2]]
            
            if C[i,j] == 1:
                ax2.plot(xs, ys, zs, '--', c = 'k')
                
            if B[i,j] == 1:
                ax2.plot(xs, ys, zs, '-', c = 'k')
    
    # Adjust view of the second example in free-standing structures to make it look better        
    if N == 12:
        ax2.view_init(10, -75)
    plt.show()

def plot_error(error_vals, title = "Error plot"):
    '''
    Make convergence plot of the error.
    '''
    N = len(error_vals)
    it = np.linspace(1, N, N, endpoint= True)
    plt.plot(it, np.log10(error_vals))
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('log10(error)')
    plt.grid(True)
    plt.show()