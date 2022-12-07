import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data file and plot

df = pd.read_csv('data.csv')

theta1d = df['Theta']
theta1d = np.array(theta1d);
theta2d = theta1d.reshape([37, 73])

phi1d = df['Phi']
phi1d = np.array(phi1d);

phi2d = phi1d.reshape([37, 73])
power1d = df['Power']

power1d = np.array(power1d);
power2d = power1d.reshape([37, 73])
THETA = np.deg2rad(theta2d)
PHI = np.deg2rad(phi2d)

R = power2d
Rmax = np.max(R)
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5, zorder=0.5)
ax.view_init(azim=300, elev=30)
plt.show()