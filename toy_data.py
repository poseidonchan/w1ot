import torch
import numpy as np
import pandas as pd
from w1ot.pytorch_implementation.ot import w1ot
from w1ot.data import make_2d_data, plot_2d_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

source, target = make_2d_data(dataset='moons', n_samples=500000, noise=0.01)
model = w1ot(source, target, 0., device)
model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=5e-4, lr_min=1e-5, optimizer='adam')
model.plot_2dpotential()
model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')

transported = model.transport(source[:1000,:], method='grad_guidance')
plot_2d_data(source[:1000,:], target[:1000,:], transported)
data = np.hstack((source[:1000,:], target[:1000,:], transported))
df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
df.to_csv('moons_1.csv', index=False)

source, target = make_2d_data(dataset='s_curve', n_samples=500000, noise=0.01)
model = w1ot(source, target, 0., device)
model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=5e-4, lr_min=1e-5, optimizer='adam')
model.plot_2dpotential()
model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')

transported = model.transport(source[:1000,:], method='grad_guidance')
plot_2d_data(source[:1000,:], target[:1000,:], transported)
data = np.hstack((source[:1000,:], target[:1000,:], transported))
df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
df.to_csv('s_curve_1.csv', index=False)

# source, target = make_2d_data(dataset='swiss_roll', n_samples=200000, noise=0.01)
# model = w1ot(source, target, 0., device)
# model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='adam')
# model.plot_2dpotential()
# model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')
#
# transported = model.transport(source[:1000,:], method='grad_guidance')
# plot_2d_data(source[:1000,:], target[:1000,:], transported)
# data = np.hstack((source[:1000,:], target[:1000,:], transported))
# df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
# df.to_csv('swiss_roll.csv', index=False)
#
# source, target = make_2d_data(dataset='circles', n_samples=200000, noise=0.01)
# model = w1ot(source, target, 0., device)
# model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='adam')
# model.plot_2dpotential()
# model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')
#
# transported = model.transport(source[:1000,:], method='grad_guidance')
# plot_2d_data(source[:1000,:], target[:1000,:], transported)
# data = np.hstack((source[:1000,:], target[:1000,:], transported))
# df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
# df.to_csv('circles.csv', index=False)

source, target = make_2d_data(dataset='checkerboard', n_samples=200000)
model = w1ot(source, target, 0., device)
model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='adam')
model.plot_2dpotential()
model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')

transported = model.transport(source[:1000,:], method='grad_guidance')
plot_2d_data(source[:1000,:], target[:1000,:], transported)
data = np.hstack((source[:1000,:], target[:1000,:], transported))
df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
df.to_csv('checkerboard.csv', index=False)

source, target = make_2d_data(dataset='8gaussians', n_samples=200000)
model = w1ot(source, target, 0., device)
model.maximize_potential(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='adam')
model.plot_2dpotential()
model.fit_distance_function(batch_size=1000, num_epochs=1000, lr_init=1e-4, lr_min=1e-5, optimizer='rmsprop')

transported = model.transport(source[:1000,:], method='grad_guidance')
plot_2d_data(source[:1000,:], target[:1000,:], transported)
data = np.hstack((source[:1000,:], target[:1000,:], transported))
df = pd.DataFrame(data, columns=['source_x', 'source_y', 'target_x', 'target_y', 'transported_x', 'transported_y'])
df.to_csv('8gaussians.csv', index=False)