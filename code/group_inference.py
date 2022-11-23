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

"""group model and guide"""
# model for group, with plate
# model shape taken from pybefit: https://github.com/dimarkov/pybefit
# this model uses Gaussian Normal distributions whose output is then mapped onto the relvant interval.
# we will look into why this is defined as it is in one of the presentations
def model_group(obs_tosses1, obs_tosses2, repetitions, n_subjects):

    npar = 2  # number of parameters: p1 and p2

    # define hyper priors over model parameters
    # prior over sigma of a Gaussian is a Gamma distribution
    a = pyro.param('a', torch.ones(npar), constraint=dist.constraints.positive)
    lam = pyro.param('lam', torch.ones(npar), constraint=dist.constraints.positive)
    tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

    sig = 1/torch.sqrt(tau)

    # each model parameter has a hyperprior defining group level mean
    # in the form of a Normal distribution
    m = pyro.param('m', torch.zeros(npar))
    s = pyro.param('s', torch.ones(npar), constraint=dist.constraints.positive)
    mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))

    # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
    # you embed what should be done for each subject into the "with pyro.plate" context
    # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
    # i.e. p1 below will have the length n_subjects
    with pyro.plate('subject', n_subjects) as ind:

        # draw parameters from Normal and transform (for numeric trick reasons)
        base_dist = dist.Normal(0., 1.).expand_by([npar]).to_event(1)
        transform = dist.transforms.AffineTransform(mu, sig)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

        # map the values in locs (between -inf and inf) onto the relevant space
        # here to between 0 and 1 using a sigmoid
        p1 = torch.sigmoid(locs[...,0])
        p2 = torch.sigmoid(locs[...,1])

        for t in pyro.markov(range(repetitions)):
            # go through trials and toss coins
            # note that in each line below, ps have length n_subject and n_subject coins will be tossed.
            coin_toss1 = pyro.sample('toss1_{}'.format(t), dist.Bernoulli(probs=p1), obs=obs_tosses1[t])
            coin_toss2 = pyro.sample('toss2_{}'.format(t), dist.Bernoulli(probs=p2), obs=obs_tosses2[t])

# guide for group, with plate
# guide shape taken from pybefit: https://github.com/dimarkov/pybefit
# this guide uses Gaussian Normal distributions whose output is then mapped onto the relvant interval.
# the guide (contrary to the model) is a multivariate normal and allows for correlations between the parameters
# we will look into why this is defined as it is in one of the presentations
def guide_group(obs_tosses1, obs_tosses2, repetitions, n_subjects):

    # number of parameters: 2, p1 and p2
    npar = 2
    trns = torch.distributions.biject_to(dist.constraints.positive)

    # define mean vector and covariance matrix of multivariate normal
    m_hyp = pyro.param('m_hyp', torch.zeros(2*npar))
    st_hyp = pyro.param('scale_tril_hyp',
                   torch.eye(2*npar),
                   constraint=dist.constraints.lower_cholesky)
    
    # set hyperprior to be multivariate normal
    hyp = pyro.sample('hyp',
                 dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                 infer={'is_auxiliary': True})

    unc_mu = hyp[..., :npar]
    unc_tau = hyp[..., npar:]

    c_tau = trns(unc_tau)

    ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
    ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

    # some numerics tricks
    mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
    tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

    m_locs = pyro.param('m_locs', torch.zeros(n_subjects, npar))
    st_locs = pyro.param('scale_tril_locs',
                    torch.eye(npar).repeat(n_subjects, 1, 1),
                    constraint=dist.constraints.lower_cholesky)

    with pyro.plate('subject', n_subjects):
        # sample unconstrained parameters from multivariate normal
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        # transform values to be between 0 and 1
        p1 = torch.sigmoid(locs[...,0])
        p2 = torch.sigmoid(locs[...,1])

        # make dictionary to be able to return samples later
        var_dict = {'p1': p1, 'p2': p2}

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

# now we can set more subjects, e.g. 3
n_subjects = 3
# they have each 2 coins which they repeatedly toss
repetitions = 30

# for better visual inspeciton of the results are the probs fixed.
probs1 = torch.tensor([0.2, 0.5, 0.8]).repeat(repetitions,1) #torch.rand(n_subjects).repeat(repetitions,1)
print("subject prob1s for coin 1:", probs1[0])
probs2 = torch.tensor([0.1, 0.7, 0.4]).repeat(repetitions,1) #torch.rand(n_subjects).repeat(repetitions,1)
print("subject prob2s for coin 2:", probs2[0])

# simulate coin tosses
obs_tosses1, obs_tosses2 = simulation(probs1, probs2, repetitions)

# set model and guide function arguments
fn_args = [obs_tosses1, obs_tosses2, repetitions, n_subjects]

# number of inference steps
iter_steps = 500

# run inference
run_svi(iter_steps, model_group, guide_group, *fn_args)

# sample from posterior
num_samples = 500
sample_df = sample_posterior(n_subjects, guide_group, *fn_args, n_samples=num_samples)

# plot the probs of coin 1 for each subject
plt.figure()
sns.displot(data=sample_df, x='p1', hue="subject", kde=True)
plt.xlim([0,1])
plt.title("p1 of each subject")
plt.show()

# plot the probs of coin 2 for each subject
plt.figure()
sns.displot(data=sample_df, x='p2', hue="subject", kde=True)
plt.xlim([0,1])
plt.title("p2 of each subject")
plt.show()