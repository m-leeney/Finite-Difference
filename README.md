# Finite-Difference
Basic Finite Difference Implementation


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.colors import ListedColormap


print("TEST 1: SCRIPT RUNNING", flush=True)



FLAME_CMAP = LinearSegmentedColormap.from_list(
    "custom_flame",
    ["#050000", "#3B0000", "#7B0000", "#b51212",
     "#ed3b00", "#ff7900", "#ffc928", "#ffff00", "#ffffff"],
    N=256,
)



#PDE Solver for the neutron diffusion equation using finite difference method
# n_t = D * (n_xx) + C*n



#Problem: 1D diffusion equation with a source term and tamper effect
def PDE_Solver_tamper(L, T, Nx, Nt, D, C, A):

    dx = (A)*(L / (Nx - 1))  # Spatial step size
    dt = T / (Nt - 1)  # Matches the spacing of the time grid below

    print("D*dt/dx**2:", D * dt / dx**2)
    

    # Initialize the solution array 
    n = np.zeros((Nt, Nx))

    # Initial condition: u(x,0) = sin(pi*x/L)
    x = np.linspace(-(A)*(L/2), (A)*(L/2), Nx)
    t = np.linspace(0, T, Nt)
    n[0, :] = 0
    n[:, -1] = 0
    n[:, 0] = 0

    

    n[0, Nx//2] = 1 / dx  # Initial condition at t=0 and x = 0
    

    C_tamper = 0.0
    C_x = np.where(np.abs(x) <= L / 2, C, C_tamper)
    

        # Time-stepping loop
    for k in range(0, Nt - 1):
        for i in range(1, Nx - 1): 
            n[k + 1, i] = n[k, i] + dt * (D * (n[k, i + 1] - 2*n[k, i] + n[k, i - 1]) / dx**2+ C_x[i] * n[k, i])
    return x, t, n



def PDE_Solver(L, T, Nx, Nt, D, C, A):

    dx = L / (Nx - 1)  # Spatial step size
    dt = T / (Nt - 1)  # Matches the spacing of the time grid below

    print("D*dt/dx**2:", D * dt / dx**2)
    



    # Initialize the solution array 
    n = np.zeros((Nt, Nx))

    # Initial condition: u(x,0) = sin(pi*x/L)
    x = np.linspace(-L/2, L/2, Nx)
    t = np.linspace(0, T, Nt)
    n[0, :] = 0
    n[:, -1] = 0
    n[:, 0] = 0

    dx=L / (Nx - 1)

    n[0, Nx//2] = 1 / dx  # Initial condition at t=0 and x = 0


    # Time-stepping loop
    for k in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            n[k + 1, i] = n[k, i] + dt * (D * (n[k, i + 1] - 2*n[k, i] + n[k, i - 1]) / dx**2+ C * n[k, i])
    return x, t, n



def exact_solution(x, t, D, C, L, dx, N_terms=500):
    n = np.zeros_like(x, dtype=float)

    for k in range(N_terms):

        m = 2*k + 1  # Only odd terms


        A_m = (2 / L) * (-1)**((m-1)//2)
        temporal = np.exp((C - D*(m*np.pi/L)**2) * t)
        spatial = np.sin((m*np.pi/L) * (x + (A)*(L/2)))

        n += A_m * temporal * spatial

    return n


print("TEST 2: CALLING PLOT FUNCTION", flush=True)

def plot_solution_3d(x, t, n, kind, heat_min=0.0, heat_max=1.0):

    print("TEST 3: INSIDE PLOT FUNCTION: {kind}, L = {length}, T = {final_time}".format(length=x[-1] - x[0], final_time=t[-1], kind=kind), flush=True)

    print("Plotting function executed")
    print("Minimum density:", np.nanmin(n))
    print("Maximum density:", np.nanmax(n))

    # Create coordinate grids
    X, T_grid = np.meshgrid(x, t)

    # Initialise figure and axes
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Determine colour scale independently of absolute maximum
    colour_max = np.nanpercentile(n, 99)

    print("99th percentile colour maximum:", colour_max)

    # Copy colormap and assign white to values above colour_max
    cmap = FLAME_CMAP.copy()
    cmap.set_over("white")

    # Nonlinear colour normalization
    norm = PowerNorm(
        gamma=0.5,
        vmin=heat_min * colour_max,
        vmax=heat_max * colour_max,
        clip=False
    )

    # Plot surface
    surface = ax.plot_surface(
        X, T_grid, n,
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        shade=False
    )

    # Colourbar
    colorbar = fig.colorbar(
        surface,
        ax=ax,
        extend="max",
        label="Neutron density, n(x,t) (99th percentile colour scale)"
    )

    colorbar.set_ticks(
        np.linspace(
            heat_min * colour_max,
            heat_max * colour_max,
            6
        )
    )

    # Axis labels
    ax.xaxis.set_pane_color((0.70, 0.70, 0.70, 1.0))
    ax.yaxis.set_pane_color((0.70, 0.70, 0.70, 1.0))
    ax.zaxis.set_pane_color((0.70, 0.70, 0.70, 1.0))    

    ax.xaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 1.0)
    ax.yaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 1.0)
    ax.zaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 1.0)

    ax.set_xlabel("Spatial Domain (x)")
    ax.set_ylabel("Time (t)")
    ax.set_zlabel("n(x,t)")

    length = x[-1] - x[0]
    final_time = t[-1]

    ax.set_title(
        f"{kind} Neutron Diffusion with Tamper: L={length:g}, T={final_time:g}, alpha={A}"
    )


    #Max Values
    print("Minimum neutron density:", np.nanmin(n))
    print("Maximum neutron density:", np.nanmax(n))


    # Save and display
    fig.savefig(f"PR{kind.lower()}_Tamper_solution_alpha={A}_L={A*length}_T={final_time}_{L}.png", dpi=300, bbox_inches="tight")

    print("SCRIPT FINISHED")

    plt.show()




# Define the parameters
L = 3.141 #45 # Length of the spatial domain. TEST: 3.0, 3.13.  3.15





#Plot T for 50, 400 and 800
T = 3000
Nx = 41
Nt = 1000000  # Small enough dt for the explicit finite-difference scheme
C = 1.0
D = 1
A = 1 #Set this to 1, if you want to ignore the tamper case.





# Solve and plot the Numerical PDE
x_num, t_num, n = PDE_Solver_tamper(L, T, Nx, Nt, D, C, A)
for k in np.linspace(0, len(t_num) - 1, 6, dtype=int):
    print(f"t={t_num[k]:.2f}, max density={np.max(n[k, :]):.6g}")
    
#x_num, t_num, n = PDE_Solver(L, T, Nx, Nt, D, C, A)
#plt.savefig('PDE_Finite_Difference_Numerical_Solution_L=3,T=40.pdf', dpi=600)




# Solve and plot the Analytical PDE
x_analytical = np.linspace(-L/2, L/2, Nx)
t_analytical = np.linspace(0.0, T, Nt)
X, T_grid = np.meshgrid(x_analytical, t_analytical)




n_exact = exact_solution(X, T_grid, D, C, L, dx=(A)*(L / (Nx - 1)), N_terms=41)

print(f"Numerical maximum at t={t_num[-1]:g}: "f"{np.nanmax(n[-1, :]):.8g}")

print(f"Analytical maximum at t={t_analytical[-1]:g}: "f"{np.nanmax(n_exact[-1, :]):.8g}")





#Numerical Plot
plot_solution_3d(x_num, t_num, n, "Numerical", heat_min=0.0, heat_max=1.0)





#Analytical Plot
# Uncomment to plot analytical results; set its heat_min/heat_max independently.
#plot_solution_3d(x_analytical, t_analytical, n_exact, "Analytical", heat_min=0.0, heat_max=1.0)

#plt.savefig('PDE_Finite_Difference_Analytical_Solution_L=3,T=40.pdf', dpi=600)
