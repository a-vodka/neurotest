from pybrain.tools.shortcuts import buildNetwork
import numpy as np

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer

from pybrain.structure import TanhLayer
from pybrain.structure import GaussianLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import SigmoidLayer

import pickle

import matplotlib.pyplot as plt


def addSamples(ds, eps, sig):
    for i in range(2, eps.size):
        deps = eps[i - 1] - eps[i - 2]
        dsig = sig[i - 1] - sig[i - 2]

        ds.addSample((eps[i - 1], sig[i - 1],
                      deps,       dsig,
                      eps[i]),
                                  sig[i])  # 1

def addSamples2(ds, eps, sig):
    for i in range(2, eps.size):
        dsig = sig[i - 1] - sig[i - 2]
        ds.addSample((eps[i - 2], eps[i - 1], dsig, eps[i]), sig[i])  # 1


eps_max, sigma_max = 35.2947, 1722310.92593

x0 = np.loadtxt("./data/Static strength - tire carcass1.TRA", dtype=float, delimiter=';', skiprows=11)
x1 = np.loadtxt("./data/Static strength - tire carcass7.TRA", dtype=float, delimiter=';', skiprows=11)
x2 = np.loadtxt("./data/Static strength - tire carcass8.TRA", dtype=float, delimiter=';', skiprows=11)

numpoint = 2000

eps_0 = x0[:numpoint, 2] / eps_max
sigma_0 = x0[:numpoint, 4] / 0.0000054 / sigma_max

eps_0 -= 0.5
sigma_0 -= 0.5

eps_1 = x1[:numpoint, 2] / eps_max
sigma_1 = x1[:numpoint, 4] / 0.0000054 / sigma_max

eps_1 -= 0.5
sigma_1 -= 0.5


eps_2 = x2[:numpoint, 2] / eps_max
sigma_2 = x2[:numpoint, 4] / 0.0000054 / sigma_max

eps_2 -= 0.5
sigma_2 -= 0.5


plt.plot(eps_0, sigma_0)
plt.plot(eps_1, sigma_1)
plt.plot(eps_2, sigma_2)
plt.show()

net = buildNetwork(4, 50, 50, 1, recurrent=False)

print net

ds = SupervisedDataSet(4, 1)

#addSamples2(ds, eps_0, sigma_0)
addSamples2(ds, eps_1, sigma_1)
#addSamples2(ds, eps_2, sigma_2)

sigma_res = np.zeros_like(eps_1)
sigma_res1 = np.zeros_like(eps_1)

trainer = BackpropTrainer(net, ds)

#trainer = RPropMinusTrainer(net, verbose = True)

results = []
result_x = []

#hl, = plt.plot([], [])
#plt.ion()
#plt.show()

def update(x, y):
    # clear
    plt.clf()
    plt.semilogy(x, y)
    # draw figure
    plt.draw()
#    plt.show()
    plt.pause(0.0001)


trainer.trainOnDataset(ds, 100)

#trainer.trainUntilConvergence(maxEpochs=100, verbose=True)
plt.ion()
for i in range(500):
    aux = trainer.train()
    results.append(aux)
    result_x.append(i)
    print aux, i
    update(result_x, results)
    if np.abs((results[i - 1] - aux) / aux) < 1e-4 and i > 2:
        break

print 'done...'

plt.close('all')
plt.ioff()
plt.close('all')
plt.ioff()


fileObject = open('a.txt', 'r+')
pickle.dump(net, fileObject)
#net = pickle.load(fileObject)
fileObject.close()

plt.semilogy(result_x, results)
plt.savefig('oshibka.png', format='png', dpi=100)
plt.show()

sigma_norm = sigma_1
eps_norm = eps_1

sigma_res[0:6] = sigma_norm[0:6]
sigma_res1[0:6] = sigma_norm[0:6]

for i in range(2, eps_norm.size):

    deps = eps_norm[i - 1] - eps_norm[i - 2]
    dsig = sigma_norm[i - 1] - sigma_norm[i - 2]
    dsig_r = sigma_res1[i - 1] - sigma_res1[i - 2]

#    sigma_res[i] = net.activate( [eps_norm[i - 1], sigma_norm[i - 1], deps, dsig, eps_norm[i]])  # 1

#    sigma_res1[i] = net.activate( [eps_norm[i - 1], sigma_res1[i - 1], deps, dsig_r, eps_norm[i]])  # 1

    sigma_res[i] = net.activate( [eps_norm[i - 2], eps_norm[i - 1], dsig, eps_norm[i]])  # 1

    sigma_res1[i] = net.activate( [eps_norm[i - 2], eps_norm[i - 1], dsig_r, eps_norm[i]])  # 1


#    if sigma_res1[i] > 1:
#        sigma_res1[i] = sigma_res1[i - 1]



# sigma_res[i] = net.activate( [eps1[i-1]-eps1[i-2], sigma1[i-1], eps1[i-2]-eps1[i-3], sigma1[i-2], eps1[i-3]-eps1[i-4], sigma1[i-3], eps1[i]-eps1[i-1]])#2
# sigma_res1[i] = net.activate( [eps1[i-1]-eps1[i-2], sigma_res1[i-1], eps1[i-2]-eps1[i-3], sigma_res1[i-2], eps1[i-3]-eps1[i-4], sigma_res1[i-3], eps1[i]-eps1[i-1]])#2

# save
# for i in range(100):
# plt.plot(eps_norm[:i], sigma_norm[:i], eps_norm[:i], sigma_res[:i], eps_norm[:i], sigma_res1[:i])

plt.plot(eps_norm, sigma_norm, eps_norm, sigma_res, eps_norm, sigma_res1)
# print sigma_res[i], sigma_res1[i], sigma_norm[i]
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig('eksperement.png', format='png', dpi=300)
plt.show()

exit()
