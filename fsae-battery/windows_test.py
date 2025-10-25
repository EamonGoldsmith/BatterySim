import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_cylinder(ax, x_offset=0):
    z = np.linspace(0, 1, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = 0.2 * np.cos(theta_grid) + x_offset
    y_grid = 0.2 * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color='skyblue', alpha=0.8)

def draw_clean_scene(n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw cylinders
    for i in range(n):
        draw_cylinder(ax, x_offset=i * 0.6)

    # Remove axes, ticks, labels
    ax.set_axis_off()

    # Optional: set equal aspect ratio
    ax.set_box_aspect([n * 0.6, 1, 1])  # X, Y, Z scaling

    plt.tight_layout()
    plt.show()

draw_clean_scene(5)
