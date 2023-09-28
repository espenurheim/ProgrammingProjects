import scipy.linalg as la
from helpFunctions import *

#Implementation of BFGS (Quasi Newton Method) #######################################
def BFGS(f, grad_f, X0, M, tol = 1.e-10, maxiter = 1000):
    """
    The BFGS-method.
    Input:
        f:          Function f
        grad_f:     Gradient of f
        X0:         Starting point
        tol:        Error tolerance
        maxiter:    Maximum number of iterations
    Output:
        X:          x-value at solution
        error_vals: Array of error values
    """

    # Initial x-value/parameters
    X = X0.copy()
    N = len(X)

    # Initial inverse matrix (identity)
    Hk = np.eye(N-3*M)

    # Find first search direction and initial error estimate
    fk_grad = grad_f(X)
    p = -fk_grad
    err = la.norm(fk_grad,2)

    #storing the error vals
    error_vals = np.array([err], dtype = float)

    # Main loop
    i = 0
    while err > tol and i < maxiter:

        # Calculate the function value and gradient
        fk = f(X)
        fk_grad = grad_f(X)

        if la.norm(fk_grad,2) > 1.e+5:
            return X.reshape(int(N/3),3), error_vals
        
        # Find step direction
        if i != 0:
            p = -Hk.dot(fk_grad)

        # Find step length using strong Wolfe conditions
        X_old = X.copy()
        fk_grad_old = fk_grad.copy()
        initial_descent = np.inner(fk_grad, p)
        X, fk, fk_grad = strongWolfe(f, grad_f, X, p, M,
                                     initial_value = fk,
                                     initial_descent = initial_descent)
        
        # Calculate new Hk
        sk = (X - X_old)[(3*M):]
        yk = fk_grad - fk_grad_old
        rho_k = 1/(np.inner(yk, sk))
        if i == 0:
            Hk = Hk * (1 / (rho_k * np.inner(yk,yk)))
        z = Hk.dot(yk)
        Hk += -rho_k * (np.outer(sk,z)+np.outer(z,sk)) + rho_k * (rho_k*np.inner(yk,z)+1) * np.outer(sk,sk)
        
        # Update error and iteration counter
        err = la.norm(fk_grad,2)
        error_vals = np.concatenate([error_vals, [err]])
        if err > 1.e+5: # If error is too large, return
            print("Error too large. Return current value.")
            return X.reshape(-1,3), error_vals
        i += 1

    print(f'# iterations = {i}')
    # Return X as a matrix to make it more readable
    return X.reshape(int(N/3),3), error_vals


#Implementation of Polak-Ribiere+ (CG method) - FOR TENSERITY-DOMES #########################
def CG(f, grad_f, X0, M, tol = 1.e-6, maxiter = 1000):
    """
    The Conjugate Gradient method.
    Input:
        f:          Function f
        grad_f:     Gradient of f
        X0:         Starting point
        C, B:       Cables and Bars
        L:          Length of cables/bars
        el:         External load on nodes
        k:          Cable elasticity parameter
        c:          Bar elasticity parameter
        M:          Number of undetermined variables
        tol:        Error tolerance
        maxiter:    Maximum number of iterations
    Output:   
        x:          x-value at solution
        error_vals: Array of error values
        
    """
    
    # Initial x-value/parameters
    X = X0.copy()
    N = len(X)

    # Find first search direction and initial error estimate
    fk_grad = grad_f(X.copy())
    p = -fk_grad

    err = la.norm(fk_grad,2)
    fk = f(X.copy())

    #storing the error vals
    error_vals = np.array([err], dtype = float)

    #main loop
    it = 0
    while err > tol and it < maxiter:
        #finding the new step using strong Wolfe conditions
        initial_descent = np.inner(fk_grad, p)
        X, fk, new_fk_grad = strongWolfe(f, grad_f, X, p, M,
                                     initial_value = fk,
                                     initial_descent = initial_descent)

        #finding the new direction
        beta_k = np.inner(new_fk_grad, (new_fk_grad - fk_grad)) / np.inner(fk_grad, fk_grad)
        #beta_k = np.inner(new_fk_grad, (new_fk_grad - fk_grad)) / np.inner(new_fk_grad - fk_grad, p)
        beta_k = np.max(beta_k, 0)
        p = - new_fk_grad + beta_k*p

        fk_grad = new_fk_grad.copy()
        err = la.norm(fk_grad,2)
                
        error_vals = np.concatenate([error_vals, [err]])
        
        # If the solution diverges, return the current solution
        if err > 1.e+5:
            print('Error too large. Returns current value.')
            return X, error_vals

        fk = f(X)

        it += 1

    print(f'# iterations = {it}')

    return X.reshape(int(N/3),3), error_vals

#Implementation of Quadratic Penalty method ###############################################
def quadraticPenalty(Q, grad_Q, X0, M, mu1 = 1000, mu2 = 10, tol = 1.e-8, maxiter = 1000):
    """
    Quadratic penalty method with BFGS.
    Input:
        Q:          objective function
        grad_Q:     objective gradient
        X0:         Starting point
        M:          Number of undetermined variables
        mu1:        Initial penalty parameter for inequality constraints
        mu2:        Initial penalty parameter for equality constraints
        tol:        Error tolerance
        maxiter:    Maximum number of iterations
    Output:   
        x:          x-value at solution
        error_vals: Array of error values
    """
    X_init = X0.copy()
    mu1_k = mu1
    mu2_k = mu2
    
    tau = np.logspace(-1, int(np.log10(tol)), int(-np.log10(tol)))
    error_vals = np.array([1])
    
    for err in tau:
        step_error = 1
        
        j = 0
        while step_error > err:
            # Define lambda functions for BFGS
            fun = lambda X: Q(X, mu1_k, mu2_k)
            grad = lambda X: grad_Q(X, mu1_k, mu2_k)
            
            # Solve with current values of mu
            X, error_vals_k = BFGS(fun, grad, X_init.flatten(), M, tol = err, maxiter = maxiter)
            step_error = error_vals_k[-1]
            iterates = len(error_vals_k)
            
            # Update X if error criteria is achieved (if not, the problem has diverged)
            if step_error < err:
                X_init = X.copy()
            
            # Update penalty parameters depending on how fast BFGS converge
            if iterates > 500: # Very slow convergence -> small increase to help with ill-conditioning
                mu1_k *= 1.5
                mu2_k *= 1.5
            else:              # fast convergence -> larger increase
                mu1_k *= 3
                mu2_k *= 3
            j += 1
            if j > 5: # If we do not converge after 5 iterations, we have a problem
                print('Problem with convergence')
                return X, error_vals
            
        error_vals = np.concatenate([error_vals, error_vals_k])
    
    return X, error_vals[1:]
            
                
                

        
        
        