import pyradi.ryplot as ryplot
import pyradi.ryprob as ryprob
import numpy as np

numpoints = 10
vec = ryprob.unifomOnSphere(numpoints)
print(vec)
print(np.linalg.norm(vec,axis=1).reshape(-1,1))

p = ryplot.Plotter(1,1,1)
p.plot3d(1, vec[:,0],vec[:,1],vec[:,2], markers='.',linewidths=[0], linestyle='-')
p.saveFig('randomonunitsphere.png')
