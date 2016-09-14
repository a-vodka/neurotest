from io import StringIO
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import time
import threading
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import matplotlib.pyplot as plt

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import TanhLayer
from pybrain.structure import GaussianLayer

import pickle

import matplotlib.pyplot as plt


def addSamples(ds, eps_norm, sigma_norm):
    for i in range(3, eps_norm.size, 5):
        ds.addSample((eps_norm[i - 1], sigma_norm[i - 1], eps_norm[i - 2], sigma_norm[i - 2], eps_norm[i - 3],
                      sigma_norm[i - 3], eps_norm[i]),
                     sigma_norm[i])  # 1


eps_max, sigma_max = 35.2947, 1722310.92593

x0 = np.loadtxt("./data/Static strength - tire carcass1.TRA", dtype=float, delimiter=';', skiprows=11)
x1 = np.loadtxt("./data/Static strength - tire carcass7.TRA", dtype=float, delimiter=';', skiprows=11)
x2 = np.loadtxt("./data/Static strength - tire carcass8.TRA", dtype=float, delimiter=';', skiprows=11)

numpoint = 1500

eps_0 = x0[:numpoint, 2] / eps_max
sigma_0 = x0[:numpoint, 4] / 0.0000054 / sigma_max

eps_1 = x0[:numpoint, 2] / eps_max
sigma_1 = x0[:numpoint, 4] / 0.0000054 / sigma_max * 0.7

eps_2 = x0[:numpoint, 2] / eps_max
sigma_2 = x0[:numpoint, 4] / 0.0000054 / sigma_max*1.3

plt.plot(eps_0, sigma_0)
plt.plot(eps_1, sigma_1)
plt.plot(eps_2, sigma_2)
plt.show()

net = buildNetwork(7, 50, 50, 1)

# print net

ds = SupervisedDataSet(7, 1)

addSamples(ds, eps_0, sigma_0)
addSamples(ds, eps_1, sigma_1)
addSamples(ds, eps_2, sigma_2)



trainer = BackpropTrainer(net, ds)

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

#trainer.trainUntilConvergence(maxEpochs=10000, verbose=True)
#plt.ion()
#for i in range(400):
#    aux = trainer.train()
#    results.append(aux)
#    result_x.append(i)
#    print aux, i
#    update(result_x, results)
#    if np.abs((results[i - 1] - aux) / aux) < 1e-5 and i > 2:
#        print 'done...'
#        plt.close('all')
#        plt.ioff()
#        break



fileObject = open('a.txt', 'r+')
#pickle.dump(net, fileObject)
net = pickle.load(fileObject)
fileObject.close()

plt.semilogy(result_x, results)
plt.savefig('oshibka.png', format='png', dpi=100)
plt.show()
plt.close()

sigma_norm = sigma_0
eps_norm = eps_0

iarr = np.arange(0, sigma_0.size, 1, dtype=float)

sigma_norm = sigma_0[iarr % 5 == 0]
eps_norm = eps_norm[iarr % 5 == 0]

sigma_res = np.zeros_like(sigma_norm)
sigma_res1 = np.zeros_like(sigma_norm)
sigma_p = np.zeros_like(sigma_norm)

sigma_res[0:6] = sigma_norm[0:6]
sigma_res1[0:6] = sigma_norm[0:6]

print eps_norm.size, sigma_norm.size, sigma_res.size, sigma_res1.size

for i in range(3, eps_norm.size):

    k = (sigma_norm[i - 2] - sigma_norm[i - 1]) / (eps_norm[i - 2] - eps_norm[i - 1])
    b = k * eps_norm[i - 1] - sigma_norm[i - 1]

    k, b = np.polyfit(eps_norm[i - 3:i], sigma_norm[i-3:i], 1)

    sigma_p[i] = k * eps_norm[i] + b


    sigma_res[i] = net.activate(
        [eps_norm[i - 1], sigma_norm[i - 1], eps_norm[i - 2], sigma_norm[i - 2], eps_norm[i - 3], sigma_norm[i - 3],
         eps_norm[i]])  # 1

    sigma_res1[i] = net.activate(
        [eps_norm[i - 1], sigma_res1[i - 1], eps_norm[i - 2], sigma_res1[i - 2], eps_norm[i - 3], sigma_res1[i - 3],
         eps_norm[i]])  # 1

    sigma_res1[i] = (0.5*sigma_res1[i] + 0.5*sigma_p[i])

#    print sigma_res[i], sigma_res1[i]

#    if sigma_res1[i] > 1:
#        sigma_res1[i] = sigma_res1[i - 1]



# sigma_res[i] = net.activate( [eps1[i-1]-eps1[i-2], sigma1[i-1], eps1[i-2]-eps1[i-3], sigma1[i-2], eps1[i-3]-eps1[i-4], sigma1[i-3], eps1[i]-eps1[i-1]])#2
# sigma_res1[i] = net.activate( [eps1[i-1]-eps1[i-2], sigma_res1[i-1], eps1[i-2]-eps1[i-3], sigma_res1[i-2], eps1[i-3]-eps1[i-4], sigma_res1[i-3], eps1[i]-eps1[i-1]])#2

# save
# for i in range(100):
# plt.plot(eps_norm[:i], sigma_norm[:i], eps_norm[:i], sigma_res[:i], eps_norm[:i], sigma_res1[:i])

plt.plot(eps_norm, sigma_norm, eps_norm, sigma_res, eps_norm, sigma_res1)
plt.plot(eps_norm, sigma_p, '--')
# print sigma_res[i], sigma_res1[i], sigma_norm[i]
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('eksperement.png', format='png', dpi=300)
plt.show()

exit()
