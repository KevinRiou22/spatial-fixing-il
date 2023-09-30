import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


batch_images = np.load("pt_cl.npy")

x = batch_images[:,:, 0].flatten()
y = batch_images[:,:, 1].flatten()
z = batch_images[:,:, 2].flatten()



fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()