import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,256)
y = x ** (1/2.2)

plt.plot(x,y)
plt.tight_layout()
plt.show()