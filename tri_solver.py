# Tridiagonal solver

import numpy as np

def tri(a,b,c,f):
    M = len(a)  
    x = np.zeros(M)  # Initialize x as a zero vector of size M
    x[0] = c[0] / b[0]  
    f[0] = f[0] / b[0]  

    # Forward sweep
    for ji in range(1, M):
        z = 1 / (b[ji] - a[ji] * x[ji-1])  
        x[ji] = c[ji] * z  # Update x
        f[ji] = (f[ji] - a[ji] * f[ji-1]) * z  

    # Backward sweep
    for ji in range(M - 2, -1, -1):  # Start from M-1 and go backward to 0
        f[ji] = f[ji] - x[ji] * f[ji + 1]  # Update f
    
    return f

