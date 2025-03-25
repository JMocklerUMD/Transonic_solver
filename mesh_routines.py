# Mesh routines

import numpy as np

def mesh_gen(mesh_inputs):
    jle, jte, jmax, kmax, xsf, ysf, kconst, dxdy = mesh_inputs
    # Set up arrays that are comparable to MATLAB original implementation
    j = np.arange(1, jmax+1)
    k = np.arange(1, kmax+1)

    xmesh = np.zeros((jmax, kmax))
    ymesh = np.zeros((jmax, kmax))

    dx1 = 1/(jte-jle+1) # Equal dx along the airfoil
    dy1 = dx1/(dxdy)

    # Initialize the mesh
    x = (j-jle-1)*dx1+0.5*dx1
    for ji in range(jle-kconst-2, -1, -1): # upstream stretching
        x[ji] = x[ji+1] + (x[ji+1]-x[ji+2])*xsf

    for ji in range(jte+kconst, jmax, 1): # downstream stretching
        x[ji] = x[ji-1] + (x[ji-1]-x[ji-2])*xsf

    y = (k-1)*dy1-0.5*dy1
    for ki in range(kconst, kmax-kconst):  # Modify for stretching in y-direction
        y[ki] = y[ki-1] + (y[ki-1] - y[ki-2]) * ysf

    for ki in range(kmax - kconst, kmax):  # Stretching in y-direction at the boundaries
        y[ki] = y[ki-1] + (y[ki-1] - y[ki-2])

    # Generate the mesh
    xmesh, ymesh = np.meshgrid(x, y, indexing='ij')

    return (xmesh, ymesh, x, y, dx1, dy1)

def diff_spacing(inputs):
    dx1, dy1, x, y, jmax, kmax = inputs
    dx = dx1*np.ones(jmax)
    for ji in range(1, jmax - 1):
        dx[ji] = 0.5*(x[ji+1] - x[ji-1])
    dx[0] = 2 * dx[1]-dx[2]
    dx[jmax-1] = 2*dx[jmax-2] - dx[jmax-3]

    dy = dy1*np.ones(kmax)
    dy[0] = y[1] - y[0]
    for ki in range(1, kmax-1):
        dy[ki] = 0.5*(y[ki+1] - y[ki-1])
    dy[kmax-1] = 2*dy[kmax-2] - dy[kmax-3]

    # Set up scaling terms for differencing
    dxp2 = np.array([1/dx[i]**2 for i in range(jmax)])
    dxm2 = np.array([1/dx[i]**2 for i in range(jmax)])

    for ji in range(1, jmax-1):
        dxp2[ji] = 1/(x[ji+1]-x[ji])/dx[ji]
        dxm2[ji] = 1/(x[ji]-x[ji-1])/dx[ji]

    dyp2 = np.array([1/dy[i]**2 for i in range(kmax)])
    dym2 = np.array([1/dy[i]**2 for i in range(kmax)])

    for ki in range(1, kmax-1):
        dyp2[ki] = 1/(y[ki+1]-y[ki])/dy[ki]
        dym2[ki] = 1/(y[ki]-y[ki-1])/dy[ki]

    return (dx, dy, dxp2, dxm2, dyp2, dym2)