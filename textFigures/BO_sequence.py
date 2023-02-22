#%%%
import numpy as np

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import synthetic

from gpytorch.kernels import MaternKernel
from gpytorch.means import ConstantMean
import matplotlib.pyplot as plt
torch.manual_seed(0)
import sys
sys.path.insert(0, "C:/Users/Usuario/Documents/Masterarbeit/code")
import tikzplotlib as tp
import warnings

warnings.filterwarnings("ignore") 

#%%%
def y(x):
    Y=0
    Y+=13*torch.cos(2 * torch.pi * x+5)
    Y+=11*torch.cos(3 * torch.pi * x+5)
    Y+=7*torch.cos(5 * torch.pi * x+5)
    Y+=5*torch.cos(7 * torch.pi * x+5)
    Y+=3*torch.cos(11 * torch.pi * x +5)
    Y+=2*torch.cos(13 * torch.pi * x+5)
    return Y
    #return (x-0.05)*(x-0.33)*(x-0.75)
def y_noisy(x, noise):
    return y(x) + (torch.normal(0,noise,x.shape))
def standardize(y):
    return (y - y.mean()) / y.std(), y.mean(), y.std()


X_true=torch.arange(start=0,end=2,step=0.01)
Y_true=y(X_true)
Y_true, m, s=standardize(Y_true) #Y_true

f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(X_true.numpy(), Y_true.numpy(), color='r')
ax.legend(['True function','Observed Data'])
plt.xlabel('input, x')
plt.ylabel('output, y')
plt.savefig('../../Writing/Figures/'+'ground_truth.png')
tp.save('../../Writing/Figures/'+'ground_truth.tex')

#%%%
X = torch.unsqueeze(torch.rand(2),1)
Y = y(X)
for i in range(20):
    train_Y=(Y-m)/s #standardize(Y)
    gp = SingleTaskGP(X, train_Y, covar_module=MaternKernel(5/2), 
                                        mean_module=ConstantMean())
    gp.covar_module.lengthscale=0.25
    gp.likelihood.noise_covar.noise=0
    #gp.covar_module.
    #print(gp)
    # print(f'lengthscale: {gp.covar_module.lengthscale}')
    # print(f'covariance: {gp.likelihood.noise_covar.raw_noise[0]}')
    # print(f'prior mean: {gp. mean_module.raw_constant}')

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    #fit_gpytorch_mll(mll);

    ##%%
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    # test model on 101 regular spaced points on the interval [0, 1]
    test_X = torch.unsqueeze(torch.linspace(0, 2, 501),1)
    # no need for gradients
    with torch.no_grad():
        # compute posterior
        posterior = gp.posterior(test_X)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
        ##Plot true function
        ax.plot(X_true.numpy(), Y_true.numpy(), color='r')
        # Plot training points as black x
        ax.plot(X.squeeze().numpy(), train_Y.numpy(), marker='x', linestyle='None', color='k')
        # Plot posterior means as blue line
        ax.plot(test_X.squeeze().numpy(), posterior.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_X.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='lightblue')
    #ax.legend(['True function','Observed Data', 'Mean', 'Confidence (2\u03C3)'])
    plt.xlabel('input, x')
    plt.ylabel('output, y')
    plt.savefig('../../Writing/Figures/'+'BO_polyn'+str(i)+'.png')
    tp.save('../../Writing/Figures/'+'BO_polyn'+str(i)+'.tex')
    plt.tight_layout()

    beta=20
    UCB = UpperConfidenceBound(gp, beta=beta)

    bounds = torch.stack([torch.zeros(1), 2*torch.ones(1)])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=200,
    )
    #print(candidate, acq_value)
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    acq=posterior.mean.numpy()+np.sqrt(beta)*np.sqrt(posterior.variance.numpy())
    ax.plot(test_X.squeeze().numpy(), acq, 'g')
    plt.xlabel('input, x')
    plt.ylabel('acquisition function, a')
    plt.savefig('../../Writing/Figures/'+'BO_polyn_acq'+str(i)+'.png')
    tp.save('../../Writing/Figures/'+'BO_polyn_acq'+str(i)+'.tex')
    plt.tight_layout()
    

    new_y=y(x=candidate)
    X=torch.cat((X,candidate))
    Y=torch.cat((Y,new_y))
    


#############refinar la acquisition function

# %%

# %%
