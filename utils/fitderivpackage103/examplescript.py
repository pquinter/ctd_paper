import numpy as np
from fitderiv import fitderiv
import matplotlib.pyplot as plt

# import data
d= np.genfromtxt('./fitderivpackage103/fig2a.csv', delimiter= ',')
t= d[:, 0]
od= d[:, 1:]

# call fitting algorithm
# try help(fitderiv) for more information
q= fitderiv(t, od, cvfn= 'sqexp', stats= True, esterrs= True)

# display results
plt.figure()
plt.subplot(2,1,1)
q.plotfit('f')
plt.subplot(2,1,2)
q.plotfit('df')
plt.show()
