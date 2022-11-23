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
def simulation(probs1, probs2, repetitions):

    tosses1 = pyro.sample('coin1', dist.Bernoulli(probs=probs1))
    tosses2 = pyro.sample('coin2', dist.Bernoulli(probs=probs2))

    return tosses1, tosses2

"""single model and guide"""
# model for single subject, without plate
def model_single(obs_tosses1, obs_tosses2, repetitions):

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=dist.constraints.positive)
    beta1 = pyro.param('beta1', torch.ones(1), constraint=dist.constraints.positive)

    p1 = pyro.sample('p1', dist.Beta(alpha1, beta1))

    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=dist.constraints.positive)
    beta2 = pyro.param('beta2', torch.ones(1), constraint=dist.constraints.positive)

    p2 = pyro.sample('p2', dist.Beta(alpha2, beta2))

    for t in pyro.markov(range(repetitions)):
        coin_toss1 = pyro.sample('toss1_{}'.format(t), dist.Bernoulli(probs=p1), obs=obs_tosses1[t])
        coin_toss2 = pyro.sample('toss2_{}'.format(t), dist.Bernoulli(probs=p2), obs=obs_tosses2[t])

# guide for single subject
def guide_single(obs_tosses1, obs_tosses2, repetitions):

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=dist.constraints.positive)
    beta1 = pyro.param('beta1', torch.ones(1), constraint=dist.constraints.positive)

    p1 = pyro.sample('p1', dist.Beta(alpha1, beta1))

    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=dist.constraints.positive)
    beta2 = pyro.param('beta2', torch.ones(1), constraint=dist.constraints.positive)

    p2 = pyro.sample('p2', dist.Beta(alpha2, beta2))

    var_dict = {'p1': p1, "p2": p2}

    return var_dict

"""group model and guide"""
# model for group, with plate

def model_group(obs_tosses1, obs_tosses2, repetitions, n_subjects):

    npar = 2  # number of parameters

    # define hyper priors over model parameters
    a = pyro.param('a', torch.ones(npar), constraint=dist.constraints.positive)
    lam = pyro.param('lam', torch.ones(npar), constraint=dist.constraints.positive)
    tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

    sig = 1/torch.sqrt(tau)

    # each model parameter has a hyperprior defining group level mean
    m = pyro.param('m', torch.zeros(npar))
    s = pyro.param('s', torch.ones(npar), constraint=dist.constraints.positive)
    mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))

    with pyro.plate('subject', n_subjects) as ind:

        base_dist = dist.Normal(0., 1.).expand_by([npar]).to_event(1)
        transform = dist.transforms.AffineTransform(mu, sig)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

        p1 = torch.sigmoid(locs[...,0])
        p2 = torch.sigmoid(locs[...,1])

        # p = pyro.sample('p', dist.Beta(alpha, beta))

        for t in pyro.markov(range(repetitions)):
            coin_toss1 = pyro.sample('toss1_{}'.format(t), dist.Bernoulli(probs=p1), obs=obs_tosses1[t])
            coin_toss2 = pyro.sample('toss2_{}'.format(t), dist.Bernoulli(probs=p2), obs=obs_tosses2[t])

def guide_group(obs_tosses1, obs_tosses2, repetitions, n_subjects):

    npar = 2
    trns = torch.distributions.biject_to(dist.constraints.positive)

    m_hyp = pyro.param('m_hyp', torch.zeros(2*npar))
    st_hyp = pyro.param('scale_tril_hyp',
                   torch.eye(2*npar),
                   constraint=dist.constraints.lower_cholesky)

    hyp = pyro.sample('hyp',
                 dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                 infer={'is_auxiliary': True})

    unc_mu = hyp[..., :npar]
    unc_tau = hyp[..., npar:]

    c_tau = trns(unc_tau)

    ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
    ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

    mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
    tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

    m_locs = pyro.param('m_locs', torch.zeros(n_subjects, npar))
    st_locs = pyro.param('scale_tril_locs',
                    torch.eye(npar).repeat(n_subjects, 1, 1),
                    constraint=dist.constraints.lower_cholesky)

    with pyro.plate('subject', n_subjects):
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        p1 = torch.sigmoid(locs[...,0])
        p2 = torch.sigmoid(locs[...,1])

        # p = pyro.sample('p', dist.Beta(alpha, beta))

        var_dict = {'p1': p1, 'p2': p2}

        return var_dict



"""inference and result handling, this is model and guide independent"""
# svi function
def run_svi(iter_steps, model, guide, *fn_args, optim_kwargs={'lr': .01},
             num_particles=10):

    pyro.clear_param_store()

    svi = pyro.infer.SVI(model=model,
              guide=guide,
              optim=pyro.optim.Adam(optim_kwargs),
              loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                              #set below to true once code is vectorized
                              vectorize_particles=True))

    loss = []
    pbar = tqdm(range(iter_steps), position=0)
    for step in pbar:#range(iter_steps):
        loss.append(torch.tensor(svi.step(*fn_args)))
        pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
        if torch.isnan(loss[-1]):
            break

    plt.figure()
    plt.plot(loss)
    plt.show()

# samples results
def sample_posterior(n_subjects, guide, *fn_args, n_samples=1000):
    # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

    p1_global = np.zeros((n_samples, n_subjects))
    p2_global = np.zeros((n_samples, n_subjects))

    for i in range(n_samples):
        sample = guide(*fn_args)
        for key in sample.keys():
            sample.setdefault(key, torch.ones(1))
        p1 = sample["p1"]
        p2 = sample["p2"]

        p1_global[i] = p1.detach().numpy()
        p2_global[i] = p2.detach().numpy()

    p1_flat = np.array([p1_global[i,n] for i in range(n_samples) for n in range(n_subjects)])
    p2_flat = np.array([p2_global[i,n] for i in range(n_samples) for n in range(n_subjects)])

    subs_flat = np.array([n for i in range(n_samples) for n in range(n_subjects)])


    sample_dict = {"p1": p1_flat, "p2": p2_flat, "subject": subs_flat}

    sample_df = pd.DataFrame(sample_dict)

    return sample_df


# n_subjects = 1
# repetitions = 20

# probs1 = torch.rand(n_subjects).repeat(repetitions,1)
# probs2 = torch.rand(n_subjects).repeat(repetitions,1)
# print("single subject probs for 1:", probs1[0], probs2[0])

# obs_tosses1, obs_tosses2 = simulation(probs1, probs2, repetitions)

# fn_args = [obs_tosses1, obs_tosses2, repetitions]

# iter_steps = 1000

# run_svi(iter_steps, model_single, guide_single, *fn_args)

# # params = pyro.get_param_store()
# # inferred_p = (params["alpha"] / (params["alpha"]+params["beta"])).detach().numpy()[0]
# # print("\ninferred p:", inferred_p)

# num_samples = 1000
# sample_df = sample_posterior(n_subjects, guide_single, *fn_args, n_samples=num_samples)
# # true_p = probs.detach().numpy()[0]

# plt.figure()
# sns.displot(data=sample_df, x='p1', hue="subject", kde=True)
# sns.displot(data=sample_df, x='p2', hue="subject", kde=True)
# # plt.plot([true_p, true_p], [0, num_samples], color = 'r', label = 'true p', scaley=False)
# # plt.plot([inferred_p, inferred_p], [0, num_samples], color = 'pink', label = 'inferred mean p', scaley=False)
# plt.xlim([0,1])
# plt.show()


n_subjects = 3
repetitions = 20

probs1 = torch.tensor([0.2, 0.5, 0.8]).repeat(repetitions,1) #torch.rand(n_subjects).repeat(repetitions,1)
print("subject prob1s for 1:", probs1[0])
probs2 = torch.tensor([0.1, 0.7, 0.4]).repeat(repetitions,1) #torch.rand(n_subjects).repeat(repetitions,1)
print("subject prob2s for 1:", probs2[0])

obs_tosses1, obs_tosses2 = simulation(probs1, probs2, repetitions)

fn_args = [obs_tosses1, obs_tosses2, repetitions, n_subjects]

iter_steps = 500

run_svi(iter_steps, model_group, guide_group, *fn_args)

# params = pyro.get_param_store()
# inferred_p = (10*torch.sigmoid(params["mu_alpha"]) / (10*torch.sigmoid(params["mu_alpha"])+10*torch.sigmoid(params["mu_beta"]))).detach().numpy()[0]
# print("\ninferred p:", inferred_p)

num_samples = 1000
sample_df = sample_posterior(n_subjects, guide_group, *fn_args, n_samples=num_samples)
# true_p = probs.detach().numpy()[0]

plt.figure()
sns.displot(data=sample_df, x='p1', hue="subject", kde=True)
plt.xlim([0,1])
sns.displot(data=sample_df, x='p2', hue="subject", kde=True)
# plt.plot([true_p, true_p], [0, num_samples], color = 'r', label = 'true p', scaley=False)
# plt.plot([inferred_p, inferred_p], [0, num_samples], color = 'pink', label = 'inferred mean p', scaley=False)
plt.xlim([0,1])
plt.show()