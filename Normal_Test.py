import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 560, 200 # mean and standard deviation
s = np.random.normal(mu, sigma, 10000)

plt.title("Samples")
plt.xlabel("Number of Customers")
plt.ylabel("%")

plt.gcf().subplots_adjust(left=0.15)
count, bins, ignored = plt.hist(s, 50, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()