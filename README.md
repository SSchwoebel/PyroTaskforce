Code and materials for pyro taskforce meetings
==============================================

Contents
--------

This repository contains a folder called 'presentations' that currently contains one 'introduction to python' presentation and will be used to upload all future presentations that are held in the scope of the taskforce.

It also contains a 'code' folder which contains... you guessed it... code. For now there is a code file which contains the code from the intoduction presentation as python code. Additionally, there are two minimal working examples, one for single subject inference, and one for group inference. These can be run and simulated data will be generated from simulated coin tosses, for which the parameters are then inferred.

Presentations
-------------

```
introduction_to_pyro.pptx
```
is a presentation that introduces pytorch (the array and numerics library behind pyro) as well as basic principles and examples of how pyro works.

Getting started
---------------

The idea is, to jointly get into pyro, and distribute and parallelize documentation reading etc.

To get started with the code, use the environment.yml file to create a conda environment which installs all the necessary requirements to run inference with pyro, as well as plot results.

First install anaconda, open a commandline of your choice, and create the environment with
```
conda env create -f environment.yml
```
and activate it using
```
conda activate PyroTaskforce
```

As an editor, the environment also installs spyder, in which you can open the python files and run them.

Code examples
-------------

```
introduction_examples.py
```
contains the code that is shown in the introduction to pyro presentation

```
single_inference.py
```
contains a minimal working example of single subject inference that infers the probability underlying simulated coin tosses.

```
group_inference.py
```
contains a minimal working example of group inference that infers probabilities simulated coins. The group inference code is from pybefit: https://github.com/dimarkov/pybefit

```
distributions.py
```
is a helper file in which the equations for distributions are implemented. It is used for plotting of the inferred distributions.
