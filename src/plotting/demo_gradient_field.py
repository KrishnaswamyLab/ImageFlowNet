import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function whose gradient we want to plot
def func(x, y, z):
    # return np.cos(x) * np.sin(y) * np.cos(z)
    return np.sin(x**2) + np.cos(z) * np.sin(y)
# Define the range and step size for each axis
x_range = np.linspace(-1, 1, 6)
y_range = np.linspace(-1, 1, 6)
z_range = np.linspace(-1, 1, 6)

# Create a meshgrid from the ranges
X, Y, Z = np.meshgrid(x_range, y_range, z_range)

# Calculate the gradient of the function at each point
grad_x, grad_y, grad_z = np.gradient(func(X, Y, Z), x_range, y_range, z_range)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vector field
ax.quiver(X, Y, Z, grad_x, grad_y, grad_z,
          length=0.12,
          color='k',
          linewidths=1.0,
          arrow_length_ratio=0.5,
          normalize=False)

# Remove grid.
ax.grid(False)
# Remove labels.
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# Remove axes lines.
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# Change box color.
ax.xaxis.set_pane_color((251/255, 229/255, 214/255, 0.4))
ax.yaxis.set_pane_color((251/255, 229/255, 214/255, 0.4))
ax.zaxis.set_pane_color((251/255, 229/255, 214/255, 0.4))

ax.set_title('Gradient Field')

plt.savefig('demo_gradient_field.png')
