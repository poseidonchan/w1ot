# W1OT: Wasserstein-1 Neural Optimal Transport Solver

## Overview

![W1OT Overview](./tree/main/Figures/fig1.png)


## Installation

```bash
# directly installation
pip install git+https://github.com/poseidonchan/w1ot.git
# Or, clone the repo and install it
# git clone https://github.com/poseidonchan/w1ot.git
# cd w1ot
# pip install -e .
```

## Usage

```python
from w1ot import w1ot, w2ot
from w1ot.data import make_2d_data, plot_2d_data

# create the toy data for training
source, target = make_2d_data(dataset='circles', n_samples=2**17, noise=0.01)
# initialize the model
model = w1ot(source, target, 0.1, device,  path='./saved_models/w1ot/circles')
# fit the Kantorovich potential
model.fit_potential_function(num_iters=10000,resume_from_checkpoint=True)
# visualize the Kantorovich potential
model.plot_2dpotential()
# fit the step size
model.fit_distance_function(num_iters=10000, resume_from_checkpoint=True)

# create the testing data
source, target = make_2d_data(dataset='circles', n_samples=2000, noise=0.01)
# apply the learned transport map
transported = model.transport(source)
# visualize the result without markers
plot_2d_data(source, target, transported, False, 0.5)

# Alternative: w2ot
# model = w2ot(source, target, 0.1, device,  path='./saved_models/w2ot/circles')
# model.fit_potential_function(num_iters=10000, resume_from_checkpoint=True)
# transported = model.transport(source)
```



## Experiments

To reproduce the experiments quickly, we suggest you install the Ray and configure your own Ray clusters. After than you can submit the experiments codes in the Experiments folder.


## Citation


## Contact

If you have any questions, feel free to email cys@umd.edu