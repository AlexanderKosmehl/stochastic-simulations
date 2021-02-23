import numpy as np
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
from scipy import optimize

# Global Variables
mu, sigma = 560, 200
overCost = 1
underCost = 10
simulationStart = 500
simulationStop = 1100
iterationsPerSimulation = 100


# Helper Functions

# Generates a sample of any size from a normal distribution
# mu = mean, sigma = standard deviation
# sampleSize = numbers to be generated, default=1
def generateGaussianSample(mu, sigma, sampleSize=1):
    sample = np.random.default_rng().normal(loc = mu, scale = sigma, size = 1)
    roundedSample = sample.round().astype(int)
    
    return roundedSample 


# Evaluates the cost of the products based on the asymetrical cost function
# Too many products = 1€ per unnescessary product
# Too few products = 10€ per missed customer
def evaluatePrintedAmount(customer, amount):
    if customer >= amount:
        return (customer - amount) * underCost
    else: 
        return (amount - customer) * overCost


# Simulates a given number of printed products any number of times and takes the average
# amount = amount of products to be printed
# times = simulation runs
def simulateAmount(amount, times=1):
    evaluations = [evaluatePrintedAmount(generateGaussianSample(mu, sigma), amount) for _ in range(0, times)]

    return average(evaluations).round().astype(int)


# Generates a list of all possible amounts of products
possibleAmounts = range(simulationStart, simulationStop)

# Simulates each of the given possible amounts
simulatedCosts = [simulateAmount(possibleAmount, iterationsPerSimulation) for possibleAmount in possibleAmounts]

# Performs a polynomial regression on the simulation results
polynomialModel = np.poly1d(np.polyfit(possibleAmounts, simulatedCosts, 3))
linespace = np.linspace(simulationStart, simulationStop-1, max(simulatedCosts))

# Get coefficients from regression model
a, b, c, d = polynomialModel.coefficients
print(f"\nPolynom: f(x) = {a} x^3 + {b} x^2 + {c} x + {d}")

# Finds minimum of regression function
min = optimize.minimize(lambda x: a*x**3 + b*x**2 + c*x + d, x0=0)
minx = round(min.x[0], 2)
miny = round(min.fun, 2)
print(f"\nMinimum of {miny} at {minx} \n")

# Plots the results
plt.title("Simulation Results")
plt.xlabel("Printed Amount")
plt.ylabel("Expected Cost")
plt.scatter(possibleAmounts, simulatedCosts)
plt.plot(linespace, polynomialModel(linespace), color="r")
plt.annotate(f"Minimum of {miny} \nat {minx}", 
                xy=(minx, miny), 
                xytext=(0.6, 0.5),
                xycoords="data", 
                textcoords="axes fraction", 
                arrowprops=dict(facecolor="black", shrink=0.05), 
                horizontalalignment="center")
plt.show()