import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameters for the Gaussian distributions
mu = np.array(
    [
        [1, -1 / np.sqrt(3), 0],
        [-1, -1 / np.sqrt(3), 0],
        [0, 2 / np.sqrt(3), 0],
        [0, 0, 4 / np.sqrt(6)],
    ]
)
cov = 0.01 * np.identity(3)

# Generate 1000 random samples from each Gaussian distribution
samples = []
for i in range(4):
    samples.append(np.random.multivariate_normal(mu[i], cov, 1000))

{
    "X": samples[i][:, 0],
    "Y": samples[i][:, 1],
    "Z": samples[i][:, 2],
}
# Plot the samples in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
colors = ["b", "b", "b", "b"]
for i in range(4):
    ax.scatter(
        samples[i][:, 0], samples[i][:, 1], samples[i][:, 2], c=colors[i], alpha=0.5
    )
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
