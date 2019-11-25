##Just to test plotting in realtime, 

import numpy as np
import matplotlib.pyplot as plt


for i in range(50):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)    
plt.show()
