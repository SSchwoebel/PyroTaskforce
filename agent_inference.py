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
# simulating Q learning agent in reversal learning task
def simulation(agent, environment, trials):
    # sample coin toss(es) from Bernoulli distribution
    outcomes = []
    actions = []
    
    for t in range(trials):
        action = agent.generate_action(t)
        outcome = environment.trial(t, action)
        agent.update_beliefs(action, outcome)
        
        outcomes.append(outcome)
        actions.append(action)


    return actions, outcomes


class Environment:
    def __init__(self, p_correct=0.8, p_change=0.0001):
        
        self.p_correct = p_correct
        self.p_change = p_change
        self.env_states = [torch.tensor(0).int()]
        self.env_probs = []

        probs_env0 = torch.tensor([[1.-p_correct, p_correct],
                                   [p_correct, 1.-p_correct]])
        probs_env1 = torch.tensor([[p_correct, 1.-p_correct],
                                   [1.-p_correct, p_correct]])
        
        self.probs = torch.stack([probs_env0, probs_env1], dim=0)
        
        
    def trial(self, t, action):
        
        change_draw = torch.rand(1)[0]
        
        if change_draw < self.p_change:            
            state = torch.logical_not(self.env_states[-1]).int()
        else:            
            state = self.env_states[-1].clone()
            
        self.env_states.append(state)
            
        curr_probs = self.probs[state, action]
        
        self.env_probs.append(self.probs[state])
        
        outcome = pyro.sample("outcome_{}".format(t), dist.Categorical(probs=curr_probs))
        
        return outcome
    

class Environment_na:
    def __init__(self, na, p_correct=0.8, p_change=0.0001):
        
        self.na = na
        self.p_correct = p_correct
        self.p_change = p_change
        self.env_states = [torch.tensor(0).int()]
        self.env_probs = []
        
        probs_base0 = torch.tensor([1.-p_correct, p_correct])
        probs_base1 = torch.tensor([p_correct, 1.-p_correct])
        
        probs_env0_list = []
        for a in range(na):
            if a%2==0:
                probs_env0_list.append(probs_base0)
            else:
                probs_env0_list.append(probs_base1)
        
        probs_env1_list = []
        for a in range(na):
            if a%2==0:
                probs_env1_list.append(probs_base1)
            else:
                probs_env1_list.append(probs_base0)
        
        probs_env0 = torch.stack(probs_env0_list, dim=0)
        probs_env1 = torch.stack(probs_env0_list, dim=0)
        
        self.probs = torch.stack([probs_env0, probs_env1], dim=0)
        
        
    def trial(self, t, action):
        
        change_draw = torch.rand(1)[0]
        
        if change_draw < self.p_change:            
            state = torch.logical_not(self.env_states[-1]).int()
        else:            
            state = self.env_states[-1].clone()
            
        self.env_states.append(state)
            
        curr_probs = self.probs[state, action]
        
        self.env_probs.append(self.probs[state])
        
        outcome = pyro.sample("outcome_{}".format(t), dist.Categorical(probs=curr_probs))
        
        return outcome

    
class Agent():
    def __init__(self, learning_rate, na=2, Q_init=torch.tensor([0,0,0,0])):
        
        self.lr = learning_rate
        self.na = na
        self.Q_init = Q_init
        
        self.Q_values = [Q_init]
        self.action_probs = [torch.exp(self.Q_init)/torch.exp(self.Q_init).sum()]
        
    def update_beliefs(self, action, outcome):
        
        Q_curr = self.Q_values[-1]
        
        a_vec = torch.eye(self.na)[:,action]
        
        Q_update = Q_curr*a_vec - self.lr*a_vec * (Q_curr*a_vec - outcome*a_vec)
        
        Q_new = torch.where(a_vec>0, Q_update, Q_curr)
        
        self.Q_values.append(Q_new)
        
        probs = torch.exp(Q_new)/torch.exp(Q_new).sum()
        
        self.action_probs.append(probs)
        
    def generate_action(self, t):
        
        action = pyro.sample("action_{}".format(t), dist.Categorical(probs=self.action_probs[-1])).int()
        
        return action
    
    def reset(self, learning_rate):
        
        self.lr = learning_rate
        self.Q_values = [self.Q_init]
        self.action_probs = [torch.exp(self.Q_init)/torch.exp(self.Q_init).sum()]
        


"""single model and guide"""
# model for single subject, without plate
def model_single(obs_actions, obs_outcomes, agent, environment, trials):

    # declare hyper parameters of the model.
    # conjugate prior of the Bernoulli is a Beta distribution.
    # it is a distribution between 0 and 1 and can therewith used to formulate a distribution over p.
    # it is used here as the prior over p for the inference.
    # it is set to be uniform below (alpha=1, beta=1).
    # alpha and beta are the co-called concentration parameters of the Beta.
    alpha = pyro.param('alpha', torch.ones(1), constraint=dist.constraints.positive)
    beta = pyro.param('beta', torch.ones(1), constraint=dist.constraints.positive)

    # sample candidate p from prior
    lr = pyro.sample('lr', dist.Beta(alpha, beta))
    
    agent.reset(lr)

    # loop through repetitions / trials
    # don't mind the pyro.markov, pyro wants this for longer loops.
    for t in pyro.markov(range(trials)):
        action = obs_actions[t]
        outcome = obs_outcomes[t]
        agent.update_beliefs(action, outcome)
        
        pyro.sample("obs_action_{}".format(t), dist.Categorical(probs=agent.action_probs[-1]), obs=action)

# guide (=posterior) for single subject
def guide_single(obs_actions, obs_outcomes, agent, environment, trials):

    # declare hyper parameters of the approximate posterior.
    # in this case, we assume a Beta again as a posterior distribution
    # we initialize uniform but pyro will update alpha and beta to fit the data
    alpha = pyro.param('alpha', torch.ones(1), constraint=dist.constraints.positive)
    beta = pyro.param('beta', torch.ones(1), constraint=dist.constraints.positive)

    # sample candidate p
    p = pyro.sample('lr', dist.Beta(alpha, beta))

    # make dict to return for sampling results later.
    var_dict = {'lr': p}

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
                              vectorize_particles=False))

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

"""
now follows the part where things are actually called and ran
"""

# only ine subject, this is single inference
trials = 100

na = 3
Q_init = torch.zeros(na)
    
outcome_dict = {"lr": [], "inf_lr": [], "inf_alpha": [], "inf_beta": []}

lrs = torch.rand([3])

for l_r in lrs:

    lr = torch.tensor([l_r])
    agent = Agent(lr, na=na, Q_init=Q_init)
    env = Environment_na(na)
    
    # simulate repeated coin tosses
    obs_actions, obs_outcomes = simulation(agent, env, trials)
    
    for k in range(na):
        plt.figure()
        plt.title("probs a{}".format(k))
        plt.plot(torch.stack(env.env_probs, dim=0)[:,k], label='true prob')
        plt.plot(torch.stack(agent.Q_values, dim=0)[:,k], label='Q_vals')
        plt.legend()
        plt.show()
    
    
    ### create new instances for inference
    
    agent_inf = Agent(0.1, na=na, Q_init=Q_init)
    env_inf = Environment_na(na)
    
    # create function arguments for model and guide
    fn_args = [obs_actions, obs_outcomes, agent_inf, env_inf, trials]
    
    # how many optimization steps we will do
    iter_steps = 500
    
    # run stochastic variational inference
    run_svi(iter_steps, model_single, guide_single, *fn_args)
    
    # get the parameters pyro has inferred
    # this gives us the inferred MEAN p
    params = pyro.get_param_store()
    inferred_lr = (params["alpha"] / (params["alpha"]+params["beta"])).detach().numpy()[0]
    inferred_alpha = params["alpha"].detach().numpy()[0]
    inferred_beta = params["beta"].detach().numpy()[0]
    # and print them
    print("\ntrue lr:", lr)
    print("inferred lr:", inferred_lr)
    print("inferred alpha:", inferred_alpha)
    print("inferred beta:", inferred_beta)
    
    outcome_dict["lr"].append(lr.numpy()[0])
    outcome_dict["inf_lr"].append(inferred_lr)
    outcome_dict["inf_alpha"].append(inferred_lr)
    outcome_dict["inf_beta"].append(inferred_lr)
    
    # # to get a distribution over p, we can sample from the posterior (guide)
    # num_samples = 500
    # sample_df = sample_posterior(n_subjects, guide_single, *fn_args, n_samples=num_samples)
    # true_p = probs.detach().numpy()[0]
    
    # # plot the sampled posterior, mean inferred p, and true p
    # plt.figure()
    # sns.displot(data=sample_df, x='p', hue="subject", kde=True, label='inferred sampled \ndistribution of p')
    # plt.plot([true_p, true_p], [0, num_samples], color = 'r', label = 'true p', scaley=False)
    # plt.plot([inferred_p, inferred_p], [0, num_samples], color = 'pink', label = 'inferred mean p', scaley=False)
    # plt.xlim([0,1])
    # plt.legend()
    # plt.title("Sampled posterior over p")
    # plt.show()
    
    # to get a distribution over p, we can also simply plug in the inferred parameters into an analytical version of the posterior
    xrange = np.arange(0,1,0.01)
    analytical_Beta = analytical_dists.Beta(xrange, alpha=inferred_alpha, beta=inferred_beta)
    
    true_lr = lr[0]
    # plot the analyitcal posterior, mean inferred p, and true p
    plt.figure()
    plt.plot([true_lr, true_lr], [0, np.amax(analytical_Beta)], color = 'r', label = 'true p', scaley=False)
    plt.plot([inferred_lr, inferred_lr], [0, np.amax(analytical_Beta)], color = 'pink', label = 'inferred mean p', scaley=False)
    sns.lineplot(x=xrange, y=analytical_Beta, label='inferred analytical\ndistribution of p')
    plt.xlim([0,1])
    plt.legend()
    plt.title("Analytical posterior over p")
    plt.show()
    
    
plt.figure()
plt.plot(outcome_dict["lr"], outcome_dict["inf_lr"], '.')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()
