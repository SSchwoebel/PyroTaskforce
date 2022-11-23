#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:06:42 2022

@author: sarah
"""

import torch
import pyro
import pyro.distributions as dist
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import distributions as analytical_dists

pyro.clear_param_store()

"""simulations"""
# simulating coin toss, can do any shape for single or group
def simulation(probs, repetitions):
    # sample coin toss(es) from Bernoulli distribution
    tosses = pyro.sample('coin', dist.Bernoulli(probs=probs))

    return tosses


"""single model and guide"""
# model for single subject, without plate
def model_single(obs_tosses, repetitions):

    # declare hyper parameters of the model.
    # conjugate prior of the Bernoulli is a Beta distribution.
    # it is a distribution between 0 and 1 and can therewith used to formulate a distribution over p.
    # it is used here as the prior over p for the inference.
    # it is set to be uniform below (alpha=1, beta=1).
    # alpha and beta are the co-called concentration parameters of the Beta.
    alpha = pyro.param('alpha', torch.ones(1), constraint=dist.constraints.positive)
    beta = pyro.param('beta', torch.ones(1), constraint=dist.constraints.positive)

    # sample candidate p from prior
    p = pyro.sample('p', dist.Beta(alpha, beta))

    # loop through repetitions / trials
    for t in pyro.markov(range(repetitions)):
        # toss a coin according to sampled p from above. Tell pyro which toss was actually observed
        coin_toss = pyro.sample('toss_{}'.format(t), dist.Bernoulli(probs=p), obs=obs_tosses[t])
        # this is used to evaluate how good different candidtae ps are.

# guide (=posterior) for single subject
def guide_single(obs_tosses, repetitions):

    # declare hyper parameters of the approximate posterior.
    # in this case, we assume a Beta again as a posterior distribution
    # we initialize uniform but pyro will update alpha and beta to fit the data
    alpha = pyro.param('alpha', torch.ones(1), constraint=dist.constraints.positive)
    beta = pyro.param('beta', torch.ones(1), constraint=dist.constraints.positive)

    # sample candidate p
    p = pyro.sample('p', dist.Beta(alpha, beta))

    # make dict to return for sampling results later.
    var_dict = {'p': p}

    return var_dict


"""inference and result handling, this is model and guide independent"""
# svi (stochastic variational inference) function
def run_svi(iter_steps, model, guide, *fn_args, optim_kwargs={'lr': .01},
             num_particles=10):

    # clear pyro just to be sure
    pyro.clear_param_store()

    # initialize SVI function and specify which model and guide to use, as well as rhe parameters.
    # adam is the optiiziation algorithm
    # also say which loss function to use, in this case trace ELBO
    # num particles says how many candidate ps we sample in each optimization step
    svi = pyro.infer.SVI(model=model,
              guide=guide,
              optim=pyro.optim.Adam(optim_kwargs),
              loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                              #set below to true once code is vectorized
                              vectorize_particles=True))

    loss = []
    pbar = tqdm(range(iter_steps), position=0)
    # loop through iteration setps. tqdm gives a little progress bar
    for step in pbar:#range(iter_steps):
        loss.append(torch.tensor(svi.step(*fn_args)))
        pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
        if torch.isnan(loss[-1]):
            break

    # plot ELBO
    plt.figure()
    plt.plot(loss)
    plt.xlabel("iter step")
    plt.ylabel("ELBO loss")
    plt.title("ELBO minimization during inference")
    plt.show()

# samples results
def sample_posterior(n_subjects, guide, *fn_args, n_samples=1000):
    # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

    p_global = np.zeros((n_samples, n_subjects))

    # sample p from guide (the posterior over p). 
    # Calling the guide yields samples from the posterior after SVI has run.
    for i in range(n_samples):
        sample = guide(*fn_args)
        for key in sample.keys():
            sample.setdefault(key, torch.ones(1))
        p = sample["p"]

        p_global[i] = p.detach().numpy()

    # do some data formatting steps
    p_flat = np.array([p_global[i,n] for i in range(n_samples) for n in range(n_subjects)])

    subs_flat = np.array([n for i in range(n_samples) for n in range(n_subjects)])


    sample_dict = {"p": p_flat, "subject": subs_flat}

    # make a pandas dataframe, better for analyses and plotting later (pandas is pythons R equivalent)
    sample_df = pd.DataFrame(sample_dict)

    return sample_df

# only ine subject, this is single inference
n_subjects = 1
# the subject throws their coin repeatedly
repetitions = 100

# sample coin toss probability and make shaped tensor out of it.
probs = torch.rand(n_subjects).repeat(repetitions,1)
print("single subject prob for 1:", probs[0])

# simulate repeated coin tosses
obs_tosses = simulation(probs, repetitions)

# create function arguments for model and guide
fn_args = [obs_tosses, repetitions]

# how many optimization steps we will do
iter_steps = 500

# run stochastic variational inference
run_svi(iter_steps, model_single, guide_single, *fn_args)

# get the parameters pyro has inferred
# this gives us the inferred MEAN p
params = pyro.get_param_store()
inferred_p = (params["alpha"] / (params["alpha"]+params["beta"])).detach().numpy()[0]
inferred_alpha = params["alpha"].detach().numpy()[0]
inferred_beta = params["beta"].detach().numpy()[0]
# and print them
print("\ninferred p:", inferred_p)
print("inferred alpha:", inferred_alpha)
print("inferred beta:", inferred_beta)

# to get a distribution over p, we can sample from the posterior (guide)
num_samples = 500
sample_df = sample_posterior(n_subjects, guide_single, *fn_args, n_samples=num_samples)
true_p = probs.detach().numpy()[0]

# plot the sampled posterior, mean inferred p, and true p
plt.figure()
sns.displot(data=sample_df, x='p', hue="subject", kde=True, label='inferred sampled \ndistribution of p')
plt.plot([true_p, true_p], [0, num_samples], color = 'r', label = 'true p', scaley=False)
plt.plot([inferred_p, inferred_p], [0, num_samples], color = 'pink', label = 'inferred mean p', scaley=False)
plt.xlim([0,1])
plt.legend()
plt.title("Sampled posterior over p")
plt.show()

# to get a distribution over p, we can also simply plug in the inferred parameters into an analytical version of the posterior
xrange = np.arange(0,1,0.01)
analytical_Beta = analytical_dists.Beta(xrange, alpha=inferred_alpha, beta=inferred_beta)

# plot the analyitcal posterior, mean inferred p, and true p
plt.figure()
plt.plot([true_p, true_p], [0, np.amax(analytical_Beta)], color = 'r', label = 'true p', scaley=False)
plt.plot([inferred_p, inferred_p], [0, np.amax(analytical_Beta)], color = 'pink', label = 'inferred mean p', scaley=False)
sns.lineplot(x=xrange, y=analytical_Beta, label='inferred analytical\ndistribution of p')
plt.xlim([0,1])
plt.legend()
plt.title("Analytical posterior over p")
plt.show()
