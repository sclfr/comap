import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 0.5  # Step dimensions (meters)
Nx, Ny = 100, 50  # Grid resolution
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Wear parameters
K = 1e-5  # Wear coefficient
F = 70 * 9.81  # Normal force (N)
L_slide = 0.1  # Sliding distance (m)
H = 7e10  # Hardness (Pa)
A_shoe = 0.014  # Shoe surface area (m^2)

# Calculate wear volume per step
V_step = K * F * L_slide / H  # Wear volume (m^3)
Delta_h = V_step / A_shoe  # Height removed per step (m)

# Footprint dimensions
foot_x, foot_y = 0.07, 0.2  # Shoe dimensions (m)
foot_dx = int(foot_x / (Lx / Nx))  # Footprint in grid points (x-direction)
foot_dy = int(foot_y / (Ly / Ny))  # Footprint in grid points (y-direction)

# Probability distribution P(x, y)
sigma_x, sigma_y = 0.12, 0.18
P_up_y = np.exp(-((Y + 0.05 * Ly)**2) / (2 * (sigma_y)**2))
P_down_y = np.exp(-((Y - 0.6 * Ly)**2) / (2 * sigma_y**2))
P_single_x = np.exp(-((X - 0.35 * Lx)**2) / (2 * sigma_x**2)) + \
             np.exp(-((X - 0.65 * Lx)**2) / (2 * sigma_x**2))
P_pairs_x = np.exp(-((X - 0.3 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.45 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.55 * Lx)**2) / (2 * sigma_x**2)) + \
            np.exp(-((X - 0.7 * Lx)**2) / (2 * sigma_x**2))

p_up, p_pairs = 0.7, 0.3
p_down, p_single = 1 - p_up, 1 - p_pairs
P_up = P_up_y * (p_single * P_single_x + p_pairs * P_pairs_x)
P_down = P_down_y * (p_single * P_single_x + p_pairs * P_pairs_x)
P = p_up * P_up + p_down * P_down
P /= np.sum(P)  # Normalize

# Parameters for time and environmental factors
T = 365 * 24 * 3600  # Total time (1 year in seconds)
n_footfalls = 10000  # Total footfalls
R = n_footfalls / T  # Footfalls per second

# Reset wear array
wear = np.zeros((Ny, Nx))
time_elapsed = 0  # Track elapsed time

# Adjusting the wear at the bottom of the step to make it far greater

# Define a modified weighting function with a larger emphasis at the bottom of the step
def enhanced_foot_pressure_weight(y_relative, y_foot_center):
    """
    Apply a significantly larger weight to the wear based on proximity to the bottom of the step.
    y_relative: y-coordinate within the foot region (relative to the foot area).
    y_foot_center: The center of the foot region in the y-direction.
    """
    base_weight = 1.0 + 2.0 * (1 - abs(y_relative - y_foot_center) / y_foot_center)
    enhanced_weight = base_weight + 5.0 * (1 - y_relative)  # Stronger emphasis near the bottom
    return enhanced_weight

# Simulate 10,000 footfalls with enhanced bottom-of-step emphasis
wear = np.zeros((Ny, Nx))  # Reset wear array

for footfall in range(1, n_footfalls + 1):
    # Time at this footfall
    time_elapsed = footfall / R
    delta_t = 1 / R  # Time since the last footfall

    # Randomly choose a footfall location based on P(x, y)
    flat_P = P.ravel()
    indices = np.random.choice(flat_P.size, p=flat_P)
    i, j = np.unravel_index(indices, P.shape)

    # Apply footfall wear with enhanced bottom emphasis
    i_start, i_end = max(0, i - foot_dy // 2), min(Ny, i + foot_dy // 2)
    j_start, j_end = max(0, j - foot_dx // 2), min(Nx, j + foot_dx // 2)

    # Handle cases where the footprint is partially out-of-bounds
    i_start = max(0, i_start)
    i_end = min(Ny, i_end)
    j_start = max(0, j_start)
    j_end = min(Nx, j_end)

    # Create a grid of relative y-values within the footprint
    foot_y_relative = np.linspace(0, 1, i_end - i_start)
    foot_weight = enhanced_foot_pressure_weight(foot_y_relative, 0.5)

    for iy in range(i_start, i_end):
        for ix in range(j_start, j_end):
            wear[iy, ix] += Delta_h * foot_weight[iy - i_start]

# Visualization
plt.imshow(wear, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
plt.colorbar(label='Wear Depth (m)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Final Wear Pattern with Strong Emphasis at Bottom of Step')
plt.show()

# Attempting an inline plot to display the 3D surface map of wear depth.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D Surface plot
surf = ax.plot_surface(X, Y, 1 - wear, cmap='viridis', edgecolor='none')

# Adding color bar and labels
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Wear Depth (m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Wear Depth (m)')
ax.set_box_aspect([8,4,1])
ax.set_title('3D Surface Plot of Wear Depth After 10,000 Footfalls')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Correcting the wear visualization to ensure the wear surface appears below the initial surface

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Initial surface (flat, no wear)
initial_wear = np.zeros_like(wear)

# Plot initial surface (red color for distinction)
ax.plot_surface(X, Y, initial_wear, color='red', alpha=0.6, edgecolor='none', label='Initial Surface')

# Plot final surface (wear is subtracted, showing below initial surface)
surf = ax.plot_surface(X, Y, -wear, cmap='viridis', alpha=0.8, edgecolor='none')

# Add color bar for the final surface
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Wear Depth (m)')

# Labels and title
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Height (m)')
ax.set_zlim(-100e-10, 0)
ax.set_box_aspect([2,1,1])
ax.set_title('3D Surface Plot: Initial (Red) vs Final Wear Depth (Below Initial Surface)')

plt.show()