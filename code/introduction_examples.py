#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 22:55:47 2022

@author: sarah
"""

import torch
import pyro

from torch.distributions import constraints
import distributions as analytical_dists
import numpy as np

def model(toss):

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    x = pyro.sample('x', pyro.distributions.Beta(alpha1, alpha2))

    pyro.sample('toss', pyro.distributions.Bernoulli(probs=x), obs=toss)




def model(tosses):

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    x = pyro.sample('x', pyro.distributions.Beta(alpha1, alpha2))

    for i in pyro.markov(range(len(tosses))):

        toss = tosses[i]

        pyro.sample('toss_'+str(i), pyro.distributions.Bernoulli(probs=x), obs=toss)


def model(data, agent):

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    lr = pyro.sample('lr', pyro.distributions.Beta(alpha1, alpha2))

    agent.reset(lr)

    for i in pyro.markov(range(data.trials)):

        stimulus = data['stimuli'][i]
        agent.update_beliefs(stimulus)

        choice_probs = agent.plan()
        choice = data['choices'][i]

        pyro.sample('choice_'+str(i), pyro.distributions.Bernoulli(probs=choice_probs), obs=choice)

def guide():

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    lr = pyro.sample('lr', pyro.distributions.Beta(alpha1, alpha2))

    params = {'alpha1': alpha1, 'alpha2': alpha2, 'lr': lr}

    return params


def guide():

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    x = pyro.sample('x', pyro.distributions.Beta(alpha1, alpha2))


def guide():

    alpha1 = pyro.param('alpha1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('alpha2', torch.ones(1), constraint=constraints.positive)

    x = pyro.sample('x', pyro.distributions.Beta(alpha1, alpha2))

    params = {'alpha1': alpha1, 'alpha2': alpha2, 'x': x}

    return params


def guide():

    alpha1 = pyro.param('beta1', torch.ones(1), constraint=constraints.positive)
    alpha2 = pyro.param('beta2', torch.ones(1), constraint=constraints.positive)

    x = pyro.sample('x', pyro.distributions.Beta(alpha1, alpha2))


def run_SVI(n_steps, n_particles):

    pyro.clear_param_store()

    svi = pyro.infer.SVI(model=model,
              guide=guide,
              optim=pyro.optim.Adam(),
              loss=pyro.infer.Trace_ELBO(num_particles=n_particles,
                              vectorize_particles=True))

    loss = []
    for step in range(n_steps):
        loss.append(torch.tensor(svi.step()))
        if torch.isnan(loss[-1]):
            break



def save_parameters(fname):

    pyro.get_param_store().save(fname)

def load_parameters(fname):

    pyro.get_param_store().load(fname)


def analytical_posterior():

    alpha1 = pyro.param('alpha1').data.numpy()
    alpha2 = pyro.param('alpha2').data.numpy()

    x_range = np.arange(0.01,1.,0.01)

    y_range = analytical_dists.Beta(x_range, alpha1, alpha2)


def sampled_posterior(n_samples):

    x_samples = np.zeros(n_samples)

    for i in range(n_samples):

        sample = guide()
        x_samples[i] = sample['x']