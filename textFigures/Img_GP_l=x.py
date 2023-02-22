#%%%

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from gpytorch.kernels import MaternKernel
from gpytorch.means import ConstantMean
import matplotlib.pyplot as plt
torch.manual_seed(0)
import sys
sys.path.insert(0, "C:/Users/Usuario/Documents/Masterarbeit/code")
import tikzplotlib as tp

#%%%
X = torch.unsqueeze(torch.rand(10),1)*(2*torch.pi)
Y = torch.sin(X)  # explicit output dimension
Y += 0.1 * (torch.randn_like(Y))
train_Y = (Y - Y.mean()) / Y.std()

X_true=torch.arange(start=0,end=(2*torch.pi),step=0.01)
Y_true=torch.sin(X_true)
Y_true=(Y_true - Y_true.mean()) / Y_true.std()


f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(X_true.numpy(), Y_true.numpy(), color='r', linestyle='dashed')
ax.plot(X.numpy(), train_Y.numpy(), marker='x', linestyle='None', color='k')
ax.legend(['True function','Observed Data'])
plt.xlabel('input, x')
plt.ylabel('output, y')
plt.savefig('../Writing/Figures/'+'Sine_ground_truth.png')
tp.save('../Writing/Figures/'+'Sine_ground_truth.tex')

#%%%
gp = SingleTaskGP(X, train_Y, covar_module=MaternKernel(5/2), 
                                    mean_module=ConstantMean())
gp.covar_module.lengthscale=2
gp.likelihood.noise_covar.noise=0.01
#gp.covar_module.
#print(gp)
print(f'lengthscale: {gp.covar_module.lengthscale}')
print(f'covariance: {gp.likelihood.noise_covar.raw_noise[0]}')
print(f'prior mean: {gp. mean_module.raw_constant}')

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#fit_gpytorch_mll(mll);

##%%
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(6, 4))
# test model on 101 regular spaced points on the interval [0, 1]
test_X = torch.unsqueeze(torch.linspace(0, (2*torch.pi), 101),1)
# no need for gradients
with torch.no_grad():
    # compute posterior
    posterior = gp.posterior(test_X)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    #Plot true function
    ax.plot(X_true.numpy(), Y_true.numpy(), color='r',linestyle='dashed')
    # Plot training points as black x
    ax.plot(X.squeeze().numpy(), train_Y.numpy(), marker='x', linestyle='None', color='k')
    # Plot posterior means as blue line
    ax.plot(test_X.squeeze().numpy(), posterior.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_X.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='lightblue')
ax.legend(['True function','Observed Data', 'Mean', 'Confidence (2\u03C3)'])
plt.xlabel('input, x')
plt.ylabel('output, y')
plt.savefig('../Writing/Figures/'+'GP_sine_l=1.png')
tp.save('../Writing/Figures/'+'GP_sine_l=1.tex')
plt.tight_layout()

# #%%%
# UCB = UpperConfidenceBound(gp, beta=0.1)


# #%%
# bounds = torch.stack([torch.zeros(1), torch.ones(1)])
# candidate, acq_value = optimize_acqf(
#     UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
# )
# print(candidate, acq_value)
# %%
