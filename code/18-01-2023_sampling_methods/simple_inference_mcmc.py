#%%
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer import MCMC, NUTS
import pandas as pd
import seaborn as sns
import torch as tr
import pyro


def model():
    """
    alpha, beta = parameters of the Beta prior of theta
    theta = coin fairness
    """

    alpha = tr.tensor(1.0)
    beta = tr.tensor(1.0)
    theta = pyro.sample("theta", dist.Beta(alpha,beta))
    return pyro.sample("outcome", dist.Bernoulli(probs=theta))

def run_inference():

    mcmc_kernel = NUTS(conditioned_model)                          # initialize proposal distribution kernel
    mcmc = MCMC(mcmc_kernel, num_samples=10000, warmup_steps=500)  # initialize MCMC class
    mcmc.run()                                                     # run inference
    return mcmc                                                    

def plot_report(mcmc):
    samples = mcmc.get_samples()                                          # extract samples of MCMC chain
    df = pd.DataFrame(data=samples['theta'].numpy(), columns=['theta'])   # create dataframe with samples, not really necessary, just personal preference
    df.hist(bins=40)                                                      # plot histogram of results
    mcmc.summary()                                                        # summarize MCMC run


def generate_data(theta, n_samples):
    """
    function for generating the data used to condition model.
    When we pass an array of parameters pyro will return an array
    with samples, where each sample was generated from a distribution
    parametrized from the corresponding parameter entry.
    """

    theta = tr.stack([theta for n in range(n_samples)])   # we want n_samples 
    return pyro.sample("y", dist.Bernoulli(probs=theta))  # each distributed according to Bernoulli(theta)



true_theta = tr.tensor(0.5)                  # true coin fairness for simulated data
n_samples = 1000                             # dataset size
data = generate_data(true_theta, n_samples)  # generate data by


# condition model on data outside of model definition
# appears to be necessary for MCMC but not sure if that is the case
conditioned_model = pyro.condition(model, data={"outcome": data})

# save a pdf of graph version of our model; need to have graphviz installed
# if on windows machine make sure Graphviz/bin is in your environment varialbes PATH
pyro.render_model(conditioned_model, filename='test.pdf')

mcmc = run_inference()
plot_report(mcmc)


