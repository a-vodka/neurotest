from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import matplotlib.pyplot as plt

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import TanhLayer
from pybrain.structure import GaussianLayer

net = buildNetwork(7, 100, 100, 1)

print net

ds = SupervisedDataSet(7, 1)


eps = np.arange(0, 0.9, 0.01)
eps2 = np.arange(0.0, 0.9, 0.01)

sigma = np.empty_like(eps)
sigma2 = np.empty_like(eps2)



last_eps = 0
last_sigma = 0
for i in range(eps.size):
    if eps[i] < 0.3:
        sigma[i] = eps[i]
    else:
        last_sigma = 0.27+0.1*eps[i]
        sigma[i] = last_sigma
        last_eps = eps[i]

for i in range(eps2.size):
        sigma2[i] = last_sigma - eps2[i]
        eps2[i] = last_eps - eps2[i]


sigma = np.append(sigma, sigma2)
eps = np.append(eps, eps2)

plt.plot(eps, sigma)
plt.show()


for i in range(eps.size - 3):
    ds.addSample( (eps[i], sigma[i], eps[i+1], sigma[i+1], eps[i+2], sigma[i+2], eps[i+3]), sigma[i+3])

trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(5000)


sigma_res = np.empty_like(eps)

for i in range(eps.size - 6):
    sigma_res[i] = net.activate( [eps[i], sigma[i], eps[i+1], sigma[i+1], eps[i+2], sigma[i+2], eps[i+3]] )

plt.plot(eps, sigma, eps, sigma_res)
plt.show()

exit()

#big
i = 0.048
x=[]
y=[]
x.append(0.012)
x.append(0.024)
x.append(0.036)
y.append(0.012)
y.append(0.024)
y.append(0.036)
k=3
ds.addSample((0,0,0,0,0,0, 0.012),(0.012))
ds.addSample((0,0,0,0,0.012,0.012, 0.024),(0.024))
ds.addSample((0,0,0.012,0.012,0.024,0.024, 0.036),(0.036))
while i <= 0.3:
    ds.addSample((x[k-3],y[k-3],x[k-2],y[k-2],x[k-1],y[k-1],i),(i))
    x.append(i)
    y.append(i)
    k=k+1
    i = i+0.012
    last=i

i = 0.3
while i <= 0.9:
    ds.addSample((x[k-3],y[k-3],x[k-2],y[k-2],x[k-1],y[k-1],i),(last))
    x.append(i)
    y.append(last)
    k=k+1
    i = i+0.024

i = 0.9
while i >= 0.3:
    ds.addSample((x[k-3],y[k-3],x[k-2],y[k-2],x[k-1],y[k-1],i),(i-0.6))
    x.append(i)
    y.append((i-0.6))
    k=k+1
    i = i-0.012
    last=(i-0.6)



print  9000




plt.plot(x,y)

plt.plot([0.1,0.2,0.3],[net.activate ([0,0,0,0,0,0, 0.1]),net.activate ([0,0,0,0,0.1,0.1, 0.2]),net.activate ([0,0,0.1,0.1, 0.2,0.2, 0.3])])

plt.show()