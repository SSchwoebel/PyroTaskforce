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
# each subject has two coins
def simulation(probs1, probs2, repetitions):

    tosses1 = pyro.sample('coin1', dist.Bernoulli(probs=probs1))
    tosses2 = pyro.sample('coin2', dist.Bernoulli(probs=probs2))

    return tosses1, tosses2

"""single model and guide"""
# model for single subject, without plate
def model_single(obs_tosses1, obs_tosses2, repetitions):

    # declare hyper parameters for coin 1
    # the prior over p1 will be a Beta again, see single_inference.py
    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=dist.constraints.positive)
    beta1 = pyro.param('beta1', torch.ones(1), constraint=dist.constraints.positive)

    # sample p1
    p1 = pyro.sample('p1', dist.Beta(alpha1, beta1))

    # declare hyper parameters for coin 2
    # the prior over p1 will be a Beta again, see single_inference.py
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=dist.constraints.positive)
    beta2 = pyro.param('beta2', torch.ones(1), constraint=dist.constraints.positive)

    # sample p2
    p2 = pyro.sample('p2', dist.Beta(alpha2, beta2))

    for t in pyro.markov(range(repetitions)):
        # in each trial the two coins are sampled
        coin_toss1 = pyro.sample('toss1_{}'.format(t), dist.Bernoulli(probs=p1), obs=obs_tosses1[t])
        coin_toss2 = pyro.sample('toss2_{}'.format(t), dist.Bernoulli(probs=p2), obs=obs_tosses2[t])

# guide for single subject
def guide_single(obs_tosses1, obs_tosses2, repetitions):

    # declare hyper parameters for coin 1
    # the prior over p1 will be a Beta again, see single_inference.py
    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=dist.constraints.positive)
    beta1 = pyro.param('beta1', torch.ones(1), constraint=dist.constraints.positive)

    # sample p1
    p1 = pyro.sample('p1', dist.Beta(alpha1, beta1))

    # declare hyper parameters for coin 2
    # the prior over p1 will be a Beta again, see single_inference.py
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=dist.constraints.positive)
    beta2 = pyro.param('beta2', torch.ones(1), constraint=dist.constraints.positive)

    # sample p2
    p2 = pyro.sample('p2', dist.Beta(alpha2, beta2))

    var_dict = {'p1': p1, "p2": p2}

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

    p1_global = np.zeros((n_samples, n_subjects))
    p2_global = np.zeros((n_samples, n_subjects))

    # sample p1 and p2 from guide (the posterior over ps). 
    # Calling the guide yields samples from the posterior after SVI has run.
    for i in range(n_samples):
        sample = guide(*fn_args)
        for key in sample.keys():
            sample.setdefault(key, torch.ones(1))
        p1 = sample["p1"]
        p2 = sample["p2"]


        p1_global[i] = p1.detach().numpy()
        p2_global[i] = p2.detach().numpy()

    # do some data formatting steps
    p1_flat = np.array([p1_global[i,n] for i in range(n_samples) for n in range(n_subjects)])
    p2_flat = np.array([p2_global[i,n] for i in range(n_samples) for n in range(n_subjects)])

    subs_flat = np.array([n for i in range(n_samples) for n in range(n_subjects)])


    sample_dict = {"p1": p1_flat, "p2": p2_flat, "subject": subs_flat}

    # make a pandas dataframe, better for analyses and plotting later (pandas is pythons R equivalent)
    sample_df = pd.DataFrame(sample_dict)

    return sample_df

"""
now follows the part where things are actually called and ran
"""

# one subject, has two coins
n_subjects = 1
# tosses each repeatedly
repetitions = 30

# create true probs for coin 1 and coin 2
probs1 = torch.rand(n_subjects).repeat(repetitions,1)
probs2 = torch.rand(n_subjects).repeat(repetitions,1)
print("single subject probs for subject 1:", probs1[0], probs2[0])

# simulate coin tosses
obs_tosses1, obs_tosses2 = simulation(probs1, probs2, repetitions)

# create function arguments for model and guide
fn_args = [obs_tosses1, obs_tosses2, repetitions]

# inference steps
iter_steps = 1000

# run inference
run_svi(iter_steps, model_single, guide_single, *fn_args)

# get the parameters pyro has inferred
# this gives us the inferred MEAN p
# for p1
params = pyro.get_param_store()
inferred_p1 = (params["alpha1"] / (params["alpha1"]+params["beta1"])).detach().numpy()[0]
inferred_alpha1 = params["alpha1"].detach().numpy()[0]
inferred_beta1 = params["beta1"].detach().numpy()[0]
# and print them
print("\ninferred p1:", inferred_p1)
print("inferred alpha1:", inferred_alpha1)
print("inferred beta1:", inferred_beta1)
# for p2
params = pyro.get_param_store()
inferred_p2 = (params["alpha2"] / (params["alpha2"]+params["beta2"])).detach().numpy()[0]
inferred_alpha2 = params["alpha2"].detach().numpy()[0]
inferred_beta2 = params["beta2"].detach().numpy()[0]
# and print them
print("\ninferred p2:", inferred_p2)
print("inferred alpha2:", inferred_alpha2)
print("inferred beta2:", inferred_beta2)

# sample from posterior
num_samples = 500
sample_df = sample_posterior(n_subjects, guide_single, *fn_args, n_samples=num_samples)

# get true probabilities
true_p1 = probs1.detach().numpy()[0]
true_p2 = probs2.detach().numpy()[0]


# plot results of p1
# plot the sampled posterior, mean inferred p1, and true p1
plt.figure()
sns.displot(data=sample_df, x='p1', hue="subject", kde=True, label='inferred sampled \ndistribution of p1')
plt.plot([true_p1, true_p1], [0, num_samples], color = 'r', label = 'true p1', scaley=False)
plt.plot([inferred_p1, inferred_p1], [0, num_samples], color = 'pink', label = 'inferred mean p1', scaley=False)
plt.xlim([0,1])
plt.legend()
plt.title("Sampled posterior over p1")
plt.show()

# to get a distribution over p1, we can also simply plug in the inferred parameters into an analytical version of the posterior
xrange = np.arange(0,1,0.01)
analytical_Beta = analytical_dists.Beta(xrange, alpha=inferred_alpha1, beta=inferred_beta1)

# plot the analyitcal posterior, mean inferred p1, and true p1
plt.figure()
plt.plot([true_p1, true_p1], [0, np.amax(analytical_Beta)], color = 'r', label = 'true p1', scaley=False)
plt.plot([inferred_p1, inferred_p1], [0, np.amax(analytical_Beta)], color = 'pink', label = 'inferred mean p1', scaley=False)
sns.lineplot(x=xrange, y=analytical_Beta, label='inferred analytical\ndistribution of p1')
plt.xlim([0,1])
plt.legend()
plt.title("Analytical posterior over p1")
plt.show()


# plot results of p2
# plot the sampled posterior, mean inferred p2, and true p2
plt.figure()
sns.displot(data=sample_df, x='p2', hue="subject", kde=True, label='inferred sampled \ndistribution of p2')
plt.plot([true_p2, true_p2], [0, num_samples], color = 'r', label = 'true p2', scaley=False)
plt.plot([inferred_p2, inferred_p2], [0, num_samples], color = 'pink', label = 'inferred mean p2', scaley=False)
plt.xlim([0,1])
plt.legend()
plt.title("Sampled posterior over p2")
plt.show()

# to get a distribution over p2, we can also simply plug in the inferred parameters into an analytical version of the posterior
xrange = np.arange(0,1,0.01)
analytical_Beta = analytical_dists.Beta(xrange, alpha=inferred_alpha2, beta=inferred_beta2)

# plot the analyitcal posterior, mean inferred p2, and true p2
plt.figure()
plt.plot([true_p2, true_p2], [0, np.amax(analytical_Beta)], color = 'r', label = 'true p2', scaley=False)
plt.plot([inferred_p2, inferred_p2], [0, np.amax(analytical_Beta)], color = 'pink', label = 'inferred mean p2', scaley=False)
sns.lineplot(x=xrange, y=analytical_Beta, label='inferred analytical\ndistribution of p2')
plt.xlim([0,1])
plt.legend()
plt.title("Analytical posterior over p2")
plt.show()