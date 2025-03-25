# Mesh routines

import numpy as np

def boundary_cnd(mesh_inputs, mesh, phi, ibc, foil_type):
    jle, jte, jmax, kmax, xsf, ysf, kconst, dxdy = mesh_inputs
    x, y, dx, dy, dxp2, dxm2, th = mesh
    bc=np.zeros(jmax)
    for ji in range(jle, jte+1):  # calculate change from interior
        
        # Return the airfoil slope for the BC
        xint = 1.008930411365
        if foil_type == 0: # NACA00XX Airfoil
            dphidy = NACA00XX_airfoil_slope(x, xint, th, ji)
        elif foil_type == 1: # Biconvex
            dphidy = biconvex_slope(x, th, ji)

        # Add additional airfoils here if desired

        velx = 0.5*((phi[ji+1, 0]-phi[ji, 0]) / dx[ji]*dxp2[ji] / dxm2[ji]+
                    (phi[ji, 0]-phi[ji-1, 0]) / dx[ji]*dxm2[ji] / dxp2[ji])
        
        if ibc == 1:
            velx = 0.0  # Use simplified boundary condition
        
        bc[ji] = -dphidy*(1+velx)*dy[0]

    return bc

def NACA00XX_airfoil_slope(x, xint, th, ji):
    slope = 5*th*(0.2969*0.5*np.sqrt(xint / x[ji])-0.126*xint-
                        0.3516*2*xint**2*x[ji]+0.2843*3*xint**3*x[ji]**2-
                        0.1015*4*xint**4*x[ji]**3)
    return slope

def biconvex_slope(x, th, ji):
    return 2*th-4*th*x[ji]