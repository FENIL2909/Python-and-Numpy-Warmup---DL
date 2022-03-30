import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

data=np.load("PoissonX.npy")
print(data.mean())
plt.subplot(2, 3, 1)
plt.hist(data,density=True)
plt.title("Emperical Probability Distribution")
plt.xlabel("Data")
plt.ylabel("Probability")
plt.xlim([0, 13])
plt.ylim([0, 0.26])

x = np.linspace(0, 12, 13)
 
# poisson distribution data for y-axis

mu= [2.5, 3.1, 3.7, 4.3]

# plotting the graph
for i in range(len(mu)):
    y = poisson.pmf(x, mu=mu[i])
    plt.subplot(2, 3, i+2)
    plt.plot(x, y)
    plt.title("Idealized Distribution with mu= %.1f" %mu[i])
    plt.xlabel("Data")
    plt.ylabel("Probability")
    plt.xlim([0, 13])
    plt.ylim([0, 0.26])
# showing the graph
plt.show()