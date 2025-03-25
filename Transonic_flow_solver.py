# ----- DESCRIPTION ------
# SLOR method for Linearized Small Disturbance (LSD) equation in 2-D
# to model compressible flow past a symmetric NACA00XX airfoil
# at zero angle of attack
# (1-M^2)phi_xx +phi_yy = 0
# on a stretched mesh, where M is the freestream Mach number

# INPUTS: Selected parameters below
# OUTPUTS: Following plots and solutions:
    # (1) The solved-over mesh
    # (2) Cp contour plot
    # (3) SLOR convergence profile
    # (4) Supersonic solution points across iterations
    # (5) Pressure contours along and above the airfoil
    # (6) Velocity perturbation field
# And more! Or whatever you choose to add

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

# Written subroutines
from tri_solver import tri
from mesh_routines import mesh_gen, diff_spacing
from airfoil_library import boundary_cnd

# -------- Make LaTeX kinda fonts -----------------
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# ----- USER INPUTS ------
# Flow/airfoil parameters
minf = 0.8   # Freestream Mach number
uinf = 1     # Freestream flow, relative to minf
gamma = 1.4  # Gas constant
th = 0.1     # Thickness
iwall = 1    # (1) wind tunnel approximation, (0) free air above
airfoil = 0  # (1) = NACA00XX airfoil, (0) = biconvex 

# Relaxation scheme parameters
omega_SLOR = 1.97    # SLOR parameter
itmax = 1000         # Iteration limit

# Meshing parameters
jle, jte = 33, 73       # leading and trailing edge locations
jmax, kmax = 105, 43    # total points in x- and y-directions
xsf, ysf = 1.2, 1.2     # stretching factors in x- and y-directions
kconst = 3              # number of constant spaced mesh points above,
dxdy=1.0                # ratio of dx to dy at airfoil surface

# Package for subroutines
mesh_inputs = (jle, jte, jmax, kmax, xsf, ysf, kconst, dxdy)


if(iwall==1):
    ysf=1.0 # Reset stretching factor in y-direction if upper wall
ibc = 0

# ------ MESH GENERATION ------
# Generate the mesh-see subroutine!
xmesh, ymesh, x, y, dx1, dy1 = mesh_gen(mesh_inputs)

# Check that everything matches!

# Set up array of central differences of mesh spacing in x-direction
cent_diff_inputs = (dx1, dy1, x, y, jmax, kmax)
dx, dy, dxp2, dxm2, dyp2, dym2 = diff_spacing(cent_diff_inputs)

# Plot the mesh!
fig1, ax1 = plt.subplots()
ax1.scatter(xmesh, ymesh, marker = ".", s = 50)

segs1 = np.stack((xmesh,ymesh), axis=2)
segs2 = segs1.transpose(1,0,2)
fig1.gca().add_collection(LineCollection(segs1))
fig1.gca().add_collection(LineCollection(segs2))
ax1.set(xlabel = "x/c, along airfoil")
ax1.set(ylabel = "y/c, normal to airfoil")
ax1.set(title = "Mesh for NACA0010 Airfoil")
ax1.set(xlim = (-0.25, 1.25))
ax1.set(ylim = (0, 1.5))

# ------ SLOR SOLVER ------
# Pre-allocate variables
a=np.zeros(kmax)
b=np.zeros(kmax)
c=np.zeros(kmax)
f=np.zeros(kmax)
cpg=np.zeros((jmax,kmax))
cp =np.zeros(jmax)
cpu=np.zeros(jmax)
res = np.zeros((jmax,kmax))

phi = np.zeros((jmax,kmax))
mu = np.zeros((jmax,kmax))
A = np.zeros((jmax,kmax))

istop = 0
l2_iter, l2res_plot = [], []
supersonic_pts = []
# Time the codeblock!!
t0 = time.time()
for it in range(1,itmax+1):

    # First calculate the boundary conditions along the airfoil
    mesh = (x, y, dx, dy, dxp2, dxm2, th)
    bc = boundary_cnd(mesh_inputs, mesh, phi, ibc, airfoil)

    # Apply bc at airfoil
    for ji in range(jmax):
        phi[ji, 0] = phi[ji, 1]+bc[ji]

    # Apply upper boundary condition
    if iwall == 1:
        for ji in range(jmax):
            phi[ji, kmax-1] = phi[ji, kmax-2]

    # Now let's calculate L_phi!
    l2res = 0
    for ki in range(1, kmax-1):  # Loop through ki
        for ji in range(2, jmax-1):  # Loop through ji
            
            # Calculate A(ji, ki)
            A[ji, ki] = 1-minf**2-minf**2*(gamma+1)*(1 / uinf)*(1 / (x[ji+1]-x[ji-1]))*(
                (phi[ji+1, ki]-phi[ji, ki])*((x[ji]-x[ji-1]) / (x[ji+1]-x[ji]))+
                (phi[ji, ki]-phi[ji-1, ki])*((x[ji+1]-x[ji]) / (x[ji]-x[ji-1])))
            
            # Set mu based on the value of A
            if A[ji, ki] >= 0:
                mu[ji, ki] = 0
            elif A[ji, ki] < 0:
                mu[ji, ki] = 1

            # X-direction contributions to residual
            res[ji, ki] = (1-mu[ji, ki])*A[ji, ki]*((phi[ji+1, ki]-phi[ji, ki]) / (x[ji+1]-x[ji])-
                                                        (phi[ji, ki]-phi[ji-1, ki]) / (x[ji]-x[ji-1]))*2 / (x[ji+1]-x[ji-1])
            
            res[ji, ki] = res[ji,ki]+mu[ji-1, ki]*A[ji-1, ki]*((phi[ji, ki]-phi[ji-1, ki]) / (x[ji]-x[ji-1])-
                                                            (phi[ji-1, ki]-phi[ji-2, ki]) / (x[ji-1]-x[ji-2]))*2 / (x[ji]-x[ji-2])

            # Y-direction contributions to residual
            res[ji, ki] = res[ji,ki]+(phi[ji, ki+1]-phi[ji, ki])*dyp2[ki]-(phi[ji, ki]-phi[ji, ki-1])*dym2[ki]

            # Accumulate the squared residuals
            l2res += res[ji, ki] ** 2


    # Calculate L2 norm of residual
    l2res = np.sqrt(l2res / (jmax*kmax))
    l2_iter.append(it)
    l2res_plot.append(l2res)

    # In addition, track the number of supersonic points
    supersonic_pts.append(mu.sum())

    # Store initial residual norm for comparison
    if it == 1:
        l2res1 = l2res

    # Check if residual has risen or dropped too much, and stop if necessary
    if l2res1 / l2res >= 10000 or l2res1 / l2res <= 1.0 / 1000 or it > 800:
        istop = 1

    # Display the residual for the current iteration
    if np.mod(it, 50) == 0:
        print(f"For Iteration {it}: L2(RES) is {l2res}")

    # Initialize dphi array for solution update
    dphi = np.zeros((jmax, kmax))

    # Use SLOR algorithm to update interior points
    for ji in range(2, jmax-1):
        # Airfoil boundary condition 
        a = np.zeros(kmax)
        b = np.zeros(kmax)
        c = np.zeros(kmax)
        f = np.zeros(kmax)
        
        a[0] = 0
        b[0] = 1
        c[0] = -1
        f[0] = 0
        
        # Interior scheme
        for ki in range(1, kmax-1):
            if A[ji, ki] >= 0:
                omega = omega_SLOR
            else:
                omega = 1.5
            
            a[ki] = dym2[ki]
            b[ki] = -(dym2[ki]+dyp2[ki])+mu[ji-1, ki]*A[ji-1, ki]*dxm2[ji-1]-(1-mu[ji, ki]*A[ji, ki])*(dxm2[ji]+dxp2[ji])
            c[ki] = dyp2[ki]
            f[ki] = -omega*res[ji, ki]-mu[ji-1, ki]*A[ji-1, ki]*(-2*omega*dphi[ji-1, ki]+omega*dphi[ji-2, ki])*dxm2[ji-1]-\
                    (1-mu[ji, ki])*A[ji, ki]*omega*dphi[ji-1, ki]*dxm2[ji]
        
        # Freestream or wall at ki = kmax
        a[kmax-1] = 0
        b[kmax-1] = 1
        c[kmax-1] = 0
        f[kmax-1] = 0
        if iwall == 1:
            a[kmax-1] = -1
        
        
        dphi[ji, :] = tri(a, b, c, f)
    
    # Update the solution!
    phi = phi+dphi

# Check how long it took
t1 = time.time()
Nsup = np.sum(mu)
del_t = t1-t0
print(f"Completed! Total CPU time {del_t}. Number of iters = {it}. Num of supersonic points = {Nsup}")

# ------ RESULTS ------
# Plot the cp curves
# Boundary condition at the leading edge (ji = 1)
ji = 0  
cp[ji] = -2*((phi[ji+1, 0]-phi[ji, 0]) / (x[ji+1]-x[ji]))
cpu[ji] = -2*((phi[ji+1, kmax-1]-phi[ji, kmax-1]) / (x[ji+1]-x[ji]))

# Loop over the interior mesh points
for ji in range(1, jmax-1):  
    cp[ji] = -2*0.5*((phi[ji+1, 0]-phi[ji, 0]) / dx[ji]*dxp2[ji] / dxm2[ji] +
                         (phi[ji, 0]-phi[ji-1, 0]) / dx[ji]*dxm2[ji] / dxp2[ji])
    cpu[ji] = -2*0.5*((phi[ji+1, kmax-1]-phi[ji, kmax-1]) / dx[ji]*dxp2[ji] / dxm2[ji] +
                          (phi[ji, kmax-1]-phi[ji-1, kmax-1]) / dx[ji]*dxm2[ji] / dxp2[ji] )

ji = jmax-1  #
cp[ji] = -2*((phi[ji, 0]-phi[ji-1, 0]) / (x[ji]-x[ji-1]))
cpu[ji] = -2*((phi[ji, kmax-1]-phi[ji-1, kmax-1]) / (x[ji]-x[ji-1]))


for ki in range(kmax): 
    ji = 0  
    cpg[ji, ki] = -2*((phi[ji+1, ki]-phi[ji, ki]) / (x[ji+1]-x[ji]))

    for ji in range(1, jmax-1): 
        cpg[ji, ki] = -2*0.5*(
            (phi[ji+1, ki]-phi[ji, ki]) / dx[ji]*dxp2[ji] / dxm2[ji] +
            (phi[ji, ki]-phi[ji-1, ki]) / dx[ji]*dxm2[ji] / dxp2[ji]
        )

    ji = jmax-1 
    cpg[ji, ki] = -2*((phi[ji, ki]-phi[ji-1, ki]) / (x[ji]-x[ji-1]))


# ------ VISUALIZATION -------
# Plot the Cp curves
fig2, ax2 = plt.subplots()
ax2.plot(x, -cp, 'bo', linestyle='-', markersize = 5, mfc='none', label = 'Numerical Cp-Surface')
ax2.plot(x, -cpu, 'r+', linestyle='-', markersize = 5, label = 'Numerical Cp-Wall')
# Uncomment to plot exact sol't of incompressible flow
#potx=np.array([0.,0.005,0.0125,0.025,0.050,0.075,0.1,0.15,0.2,0.25, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.])
#potcp=np.array([1.,0.282,-.061,-.237,-.325,-.341,-.341,-.341,-.329,-.309, -.284,-.237,-.190,-.138,-.094,-.040,.04,.075,1.])
#ax2.plot(potx, -potcp, 'gx', markersize = 8, label = 'Incompressible Exact')

ax2.set(ylabel = "-Cp")
ax2.set(xlabel = "x/c, along the airfoil")
ax2.set(title = "Cp Along the Surfaces")
ax2.set(xlim = (-0.25, 1.25))
ax2.set(ylim = (-1.5, 1))
ax2.grid(True)
ax2.legend()

# Plot the l2 residual curves
fig4, ax4 = plt.subplots()
ax4.semilogy(l2_iter, l2res_plot, 'b.', linestyle='-', markersize = 5, mfc='none')
ax4.set(ylabel = "L2 Residual")
ax4.set(xlabel = "Iteration Count")
ax4.set(title = "SLOR Convergence")
ax4.grid(True)

# Plot the supersonic points counter
fig5, ax5 = plt.subplots()
ax5.plot(l2_iter, supersonic_pts, 'bo', linestyle='--', markersize = 5, mfc='none')
ax5.set(ylabel = "Number of Supersonic Points")
ax5.set(xlabel = "Iteration Count")
ax5.set(title = "Supersonic Points in the Solution")
ax5.grid(True)

# Phi contour
fig7, ax7 = plt.subplots()
ax7.contour(xmesh, ymesh, cpg, levels = 40)
ax7.set(ylabel = "y/c, normal to the airfoil")
ax7.set(xlabel = "x/c, along the airfoil")
ax7.set(title = "Pressure Contours over NACA0010")
ax7.set(xlim = (-0.25, 1.25))
ax7.set(ylim = (-0.25, 1.25))
ax7.grid(True)

# Velocity profile contour
x_vector_field = x[jle-3:jte+3:2]
y_vector_field = y[0:16:2]
X, Y = np.meshgrid(x_vector_field, y_vector_field, indexing='ij')

# Take derivatives
Ufull = np.zeros((jmax, kmax))
Vfull = np.zeros((jmax, kmax))
for ji in range(1,jmax-1):
    for ki in range(1, kmax-1):
        Ufull[ji, ki] = (phi[ji+1,ki]-phi[ji-1,ki])/(2*dx1)
        Vfull[ji, ki] = (phi[ji,ki+1]-phi[ji,ki-1])/(2*dy1)

U = Ufull[jle-3:jte+3:2, 0:16:2]
V = Vfull[jle-3:jte+3:2, 0:16:2]

fig6, ax6 = plt.subplots()
ax6.quiver(X, Y, U, V)
ax6.set(ylabel = "y/c, normal to the airfoil")
ax6.set(xlabel = "x/c, along the airfoil")
ax6.set(title = "Velocity Perturbation Profile over NACA0010")
ax6.set(xlim = (0, 1))
ax6.set(ylim = (0, 1))
ax6.grid(True)

# Output the plots
plt.show()