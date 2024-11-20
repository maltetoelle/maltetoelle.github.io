---
title: "Bayesian Neural Networks with Variational Inference"
date: 2023-05-13 15:58:00 +0100
permalink: /posts/2023/13/bayesian-neural-networks-with-variational-inference/
categories: linear regression vi mcmc
---

The general concept of variational inference (VI) and one application to linear regression can be found <a href="">here</a>. We will quickly review the basic concepts needed to understand the following.

In contrast to maximum likelihood, which assumes one true parameter set $$\hat{\boldsymbol{\theta}}$$ that best explains our data, the Bayesian approach imposes a prior distribution onto the parameters and thereby treating them as random variables. A comparison for the simplest case of linear regression can be found <a href="https://maltetoelle.github.io/linear/regression/2020/10/27/try.html">here</a>. For a data set $$\mathcal{D}=(\mathbf{x}_i,\mathbf{y}_i)$$ Bayes theorem is given by

$$p(\boldsymbol{\theta}|\mathcal{D})=\frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})}=\frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{\int p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})\,d\boldsymbol{\theta}}~,$$

where the posterior of our model parameters $$p(\boldsymbol{\theta}\vert\mathcal{D})$$ is obtained by multiplying the likelihood $$p(\mathcal{D}\vert\boldsymbol{\theta})$$, the probability for seeing this data with our model parameters, with the prior $$p(\boldsymbol{\theta})$$, our assumption for the parameter distribution of the parameters before seeing any data.

This can also be an uniform (non-informative) prior, assigning the same probability to all parameter distributions. To obtain a valid probability distribution the product of the two must be normalized to integrate to one, which is done by dividing by the evidence $$p(\mathcal{D})$$ that is obtained by marginalizing out all possible parameter distributions. Even for simple models that are non-linear this calculation becomes intractable. So we must help ourselves with approximation frameworks such as variational inference or Markov Chain Monte Carlo (MCMC) sampling, which was compared to linear regression with variational inference in the same <a href="">post</a>.

The predictive distribution for a new data point $$(\mathbf{x}_*,\mathbf{y}_*)$$ is obtained in the Bayesian framework by marginalizing out the posterior parameter distribution

$$p(\mathbf{y}_*|\mathbf{x}_*,\mathcal{D})=\int p(\mathbf{y}_*|\mathbf{x}_*,\mathcal{D},\boldsymbol{\theta})p(\boldsymbol{\theta}|\mathcal{D})\,d\boldsymbol{\theta}~.$$

In VI we use a simpler distribution $$q(\boldsymbol{\theta})$$ to approximate our intractable posterior. The optimization objective is then given by the minimum of the KL divergence between approsimate and true posterior

$$F(q):=\mathrm{KL}(q(\boldsymbol{\theta})||p(\boldsymbol{\theta}|\mathcal{D}))=\int q(\boldsymbol{\theta})\log \frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta}|\mathcal{D})}\,\mathrm{d}\boldsymbol{\theta} \longrightarrow \underset{q(\boldsymbol{\theta}) \in \mathcal{Q}}{\min} ~.$$

Although it is not a true distance measure because of its asymmetry, it can be seen as one, as it has its minimum of zero, if and only if both distributions are equal. For all other distributions it is always greater than zero. Befor we dive deeper into the derivations for applying VI to neural networks, we will quickly revisit the important properties of the KL divergence, as they come in handy later.

## KL Divergence

The KL divergence is defined as (<a href="https://projecteuclid.org/download/pdf_1/euclid.aoms/1177729694">Kullback and Leibler 1951</a>)

$$
\begin{align}
\mathrm{KL}(q(x)||p(x)) &= \textrm{H}(q(x),p(x)) - \textrm{H}(q(x)) \\
&= - \int q(x) \log p(x)\,dx - \left( - \int q(x)\log q(x)\,dx \right) \\
&= \int q(x)\log\frac{q(x)}{p(x)}\,dx~,
\end{align}
$$

where $$\textrm{H}(q(x),p(x))$$ denotes the cross-entropy between $$q$$ and $$p$$ and $$\textrm{H}(q(x))$$ is the entropy of $$q$$. Formally, the KL divergence measures, how much information is lost, when p is approximated by q or vice versa. For two Gaussian distributions the KL divergence can be computed analytically

$$
\begin{aligned}
\mathrm{KL}(q(x)||p(x)) &= \int q(x)\log q(x)\,dx - \int q(x)\log p(x)\,dx\\
&= \frac{1}{2}\log\left(2\pi\sigma_p^2\right) + \frac{\sigma_q^2 + \left( \mu_q - \mu_p \right)^2}{2\sigma_p^2} - \frac{1}{2}\left( 1 + \log\left(2\pi\sigma_q^2\right) \right)\\
&= \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + \left( \mu_q - \mu_p \right)^2}{2\sigma_p^2} - \frac{1}{2} ~.
\end{aligned}
$$

For more complicated distributions, where the KL divergence is not analytically tractable, but the expectation can be approximated using Monte Carlo (MC) samples:

$$
\mathrm{KL}(q(x)||p(x)) = \mathbb{E}_{x\sim q}\left[ \log\frac{q(x)}{p(x)} \right] \approx \sum_{i=0}^{N}\left( \log q(x_i) - \log p(x_i) \right)~.
$$

An example for a more complicated distribution is the mixture of Gaussians:

$$p(x)=\sum_i \pi_i \mathcal{N}(x|\mu_i,\sigma_i)~.$$

The important properties of the KL divergence are:

$$
\begin{aligned}
    \textrm{non-negativity}&: \quad \mathrm{KL}(q(x)||p(x)) \geq 0 ~, \;\; \forall x ~,\\
    \textrm{equality}&: \quad \mathrm{KL}(q(x)||p(x)) = 0 \quad \textrm{if and only if} \quad q(x) = p(x) ~,\\
   \textrm{asymmetry}&: \quad \mathrm{KL}(q(x)||p(x)) \neq \mathrm{KL}(p(x)||q(x)) ~.
\end{aligned}
$$

The KL divergence becomes zero, if and only if both distributions are equal. For all other distributions it is always greater than zero. As already said it is not a true distance metric becaus of its asymmetry. We distinguish between the reverse (exclusive) $$\mathrm{KL}(q(x)\vert\vert p(x))$$ and the forward (inclusive) $$\mathrm{KL}(p(x)\vert\vert q(x))$$ case. The following illustrates both cases.


```python
import torch
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

def mc_kl_divergence(p: Distribution, q: Distribution, n_samples: int = 1) -> torch.Tensor:
    kl = 0
    for _ in range(n_samples):
        sample = p.rsample()
        kl += p.log_prob(sample) - q.log_prob(sample)
    return kl / n_samples

class MixtureNormal(Distribution):
    def __init__(self, loc, scale, pi):
        super(MixtureNormal, self).__init__()

        assert(len(loc) == len(pi))
        assert(len(scale) == len(pi))

        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        self.pi = torch.tensor(pi)

        self.dists = [Normal(loc, scale) for loc, scale in zip(self.loc, self.scale)]

    def rsample(self) -> torch.Tensor:
        x = torch.rand(1)
        rsample = 0
        for pi, dist in zip(self.pi, self.dists):
            rsample += pi * torch.exp(dist.log_prob(dist.cdf(x)))
        return rsample

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pdf = 0
        for pi, dist in zip(self.pi, self.dists):
            pdf += pi.to(x.device) * torch.exp(dist.log_prob(x))
        return torch.log(pdf)
```


```python
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

def optimize_kl(p: Distribution, intial_mu_q: float = 0., intial_sigma_q: float = 1., kl_type: str = 'forward',
                num_iter: int = 100, n_samples: int = 100) -> (float, float):

    # intial values for mu_q and sigma_q
    mu_q = torch.tensor([intial_mu_q], requires_grad=True)
    sigma_q = torch.tensor([intial_sigma_q], requires_grad=True)
    optim = torch.optim.Adam([mu_q, sigma_q])
    for i in tqdm(range(num_iter)):
        optim.zero_grad()
        q = Normal(mu_q, sigma_q)
        if kl_type == 'forward':
            loss = mc_kl_divergence(q, p, n_samples=n_samples)
        else:
            loss = mc_kl_divergence(p, q, n_samples=n_samples)
        loss.backward()
        optim.step()
    return mu_q.item(), sigma_q.item()
```


```python
mu1, mu2, sigma1, sigma2 = 0, 3, 0.4, 1.
x = torch.linspace(-4, 7, 1000)

p = MixtureNormal(loc=[mu1, mu2], scale=[sigma1, sigma2], pi=[0.5, 0.5])

fig, axs = plt.subplots(1, 2, figsize=(15,5))
for i, kl_type in enumerate(['reverse', 'forward']):
    mu_q, sigma_q = optimize_kl(p, intial_mu_q=1.5, intial_sigma_q=2., kl_type=kl_type, num_iter=1000, n_samples=50)
    q = Normal(mu_q, sigma_q)

    axs[i].plot(x, p.log_prob(x).exp(), label=r'$p(x)$')
    axs[i].plot(x, q.log_prob(x).exp(), label=r'$q(x)$')
    axs[i].set_xlabel(r'$x$', fontsize=17)
    axs[i].set_ylabel(r'$p(x)$', fontsize=17)
    axs[i].set_title(kl_type, fontsize=22)
    axs[i].legend()

plt.show()
```

    100%|██████████| 1000/1000 [00:37<00:00, 26.65it/s]
    100%|██████████| 1000/1000 [00:30<00:00, 32.34it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_3_1.png)



To better understand the two behaviours we brake down the KL divergence piece by piece. As can be seen in the reverse case the KL divergence approximates the mode of the $$p$$ and thereby leaving a lot of mass uncovered. As we would like to minimize the KL divergence we must minimize the $$\log\frac{q}{p}$$ term, which is small in areas, where $$p$$ is large. Thereby the objective converges to a mode of $$p$$. Consequently, in the forward case on the other hand the objective gets small in areas, where $$p$$ is small.

## (Log) Evidence Lower Bound

One can rewrite the log evidence such that we are left with the KL divergence from above and another term, named the (log) evidence lower bound or short ELBO

$$
\begin{aligned}
\log p(\mathcal{D}) &= \int q(\boldsymbol{\theta})\log\frac{p(\mathcal{D},\boldsymbol{\theta})}{q(\boldsymbol{\theta})}\,d\boldsymbol{\theta} + \int q(\boldsymbol{\theta})\frac{q(\boldsymbol{\theta})}{p(\boldsymbol{\theta}|\mathcal{D})}\,d\boldsymbol{\theta} \\
&= \textrm{ELBO}(q(\boldsymbol{\theta})) + \mathrm{KL}(q(\boldsymbol{\theta})||p(\boldsymbol{\theta}|\mathcal{D})) \\
&\geq \textrm{ELBO}(q(\boldsymbol{\theta})) ~.
\end{aligned}
$$

As the KL divergence is always greater than zero, maximizing the first term, the ELBO, w.r.t. $$q$$ is essentially equal to minimizing the second term, the KL divergence.



When applying the ELBO as optimization criterion to neural networks it must further be simplified into a data-dependent likelihood term and a regularizer measuring the "distance" between prior and posterior.

$$
\begin{aligned}
\textrm{ELBO}(q(\boldsymbol{\theta})) &= \int q(\boldsymbol{\theta}) \log \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{q(\boldsymbol{\theta})}\,d\boldsymbol{\theta}  \\
&= \int q(\boldsymbol{\theta}) \log p(\mathcal{D}|\boldsymbol{\theta})\,d\boldsymbol{\theta} + \int q(\boldsymbol{\theta})\log\frac{p(\boldsymbol{\theta})}{q(\boldsymbol{\theta})}\,d\boldsymbol{\theta} \\
&= \underbrace{\mathbb{E}_{\boldsymbol{\theta}\sim q} \log p(\mathcal{D}|\boldsymbol{\theta})}_{\textrm{likelihood term}} - \underbrace{\mathrm{KL}(q(\boldsymbol{\theta})||p(\boldsymbol{\theta}))}_{\textrm{regularizer}}
\end{aligned}
$$

In each optimization step we sample weights $$\boldsymbol{\theta}$$ from our approximate posterior $$q(\boldsymbol{\theta})$$. The expectation of the likelihood term then measures, how well on average our model fits the data. The KL divergence measures how well our approximate posterior matches our prior. Since it must be minimized in order to minimize the ELBO, the model tries to keep the posterior as close as possible to the prior. For computing the above KL divergence we can now either use the forward or reverse version.

Now, the question remains to which family of distributions $$\mathcal{Q}$$ to restrict $$q$$ to allow for tractable soluitons for approximating the true posterior. We can either use a parametric distribution $$q_{\boldsymbol{\omega}}(\boldsymbol{\theta})$$ governed by a set of parameters $$\boldsymbol{\omega}$$. Hence, the ELBO becomes a function of $$\boldsymbol{\omega}$$, and we can exploit standard non-linear optimization techniques to determine the optimal values for the parameters. Another possibility is to use factorized distributions. We will revisit both concepts in the domain of neural networks and apply them to the case of non-linear regression.

## Mean Field Assumption

As was the case in the linear regression example we start by using a factorized distribution as approximation, known as mean field approximation

$$q(\boldsymbol{\theta})=\prod_i q_i(\boldsymbol{\theta}_i)~,$$

which discards covariances in the parameters because of the factorization, but leads to faster computation time as this decreases the number of parameters to optimize as well. To model each individual $$q_i(\boldsymbol{\theta}_i)$$ we will use a Gaussian distribution $$\mathcal{N}(\boldsymbol{\theta}_i\vert\boldsymbol{\mu}_i,\boldsymbol{\sigma}_i)$$ as this is justified under the Bayesian central limit theorem. As can be seen in the comparison of MCMC sampling and VI for linear regression this assumption usually leads to an underestimation of the variance in the parameters.

When we want to apply the mean field approximation to a neural network a problem arises, when we want to apply the backpropagation algorithm to a probability distribution. The general update formula of gradient descent, which lies at the heart of backpropagation, is given by

$$\theta_{ij}^* = \theta_{ij} - \eta \frac{\partial\mathcal{L}(\boldsymbol{\theta})}{\partial\theta_{ij}}~,$$

where the new value $$\theta_{ij}^*$$ for each weight after every iteration is obtained by subtracting the partial derivate of some loss function $$\mathcal{L}(\boldsymbol{\theta})$$ (e.g. mean squared error, cross-entropy) w.r.t. that particular weight $$\theta_{ij}$$ weighted by a learning rate $$\eta$$ from $$\theta_{ij}$$. The above formula can be applied to point estimates of the parameters only, each weight must have one explicitly defined value. Following from that, it is not applicable to probability distributions in their standard form.

### (Local) Reparametrization Trick

Remedy comes in the form of the reparametrization trick, which separates the deterministic and stochastic components of the weights (<a href="https://arxiv.org/abs/1312.6114">Kingma and Welling 2013</a>, <a href="https://arxiv.org/abs/1505.05424">Blundell et al. 2015</a>). Instead of sampling weights directly from $$q(\boldsymbol{\theta})$$ the mean and the variance of the Gaussians modelling the weights are treated as parameters and another random variable $$\boldsymbol{\epsilon}$$ is introduced:

$$
\begin{gathered}
\theta_{ij} \sim \mathcal{N}(\mu_{ij},\sigma_{ij}) ~,\\
\theta_{ij} = \mu_{ij} + \sigma_{ij}\epsilon_{ij} \quad \textrm{with} \quad \epsilon_{ij} \sim \mathcal{N}(0,1) ~.
\end{gathered}
$$

In each forward pass the weights are sampled according to the formula above and then, subsequently, the partial derivate w.r.t. to mean and variance is computed. This essentially means the number of parameters is doubled. To ensure an always positive variance, the reparametrization is usually extended with the Softplus function:

$$\theta_{ij}=\mu_{ij} + \log\left( 1+ \exp\left(\sigma_{ij}^2\right) \right)\epsilon_{ij}~.$$

The following images explain the induced differences of using reparametrization visually. On the left the initial situation is presented, where each weight $$\theta$$ is modelled with a distribution $$q$$. On the right the reparametrization trick is shown, in which in each forward pass a weight is sampled.

<div>
  <img src="/assets/imgs/BNNs_with_VI_files/usual-1.png" width="200"/>
  <img src="/assets/imgs/BNNs_with_VI_files/rt-1.png" width="200"/>
</div>

The reparametrization trick described above still exhibits limitations concerning the variance. If we were to sample one weight for each mini-batch, the resulting outputs would show high covariances. We could circumvent this problem by sampling a separate weight for each sample in the mini-batch, but this is computational expensive. <a href="https://arxiv.org/abs/1506.02557">Kingma et al. (2015)</a> first discovered that for a factorized Gaussian posterior on the weights, the posterior on the activations is also a factorized Gaussian. Thus, instead of sampling the weights directly we can also sample from the pre-activation neuron. <a href="https://arxiv.org/abs/1506.02557">Kingma et al. (2015)</a> report much lower variance and computational time for their gradient estimator termed local reparametrization trick. We will conduct a comparison later, when we have implemented both. More formally, their reparametrization trick is mathematically given by

$$
\begin{gathered}
q_{\omega}(\theta_{ij})=\mathcal{N}(\mu_{ij},\sigma_{ij}^{2}) \;\; \forall \;\; \theta_{ij} \in \boldsymbol{\theta} \quad \Longrightarrow \quad q_{\omega}(b_{mj}|\mathbf{A})=\mathcal{N}(\gamma_{mj},\delta_{mj}) ~,\\
\gamma_{mj}=\sum_{i} a_{mi}\mu_{ij} ~,\\
\delta_{mj}=\sum_{i} a_{mi}^{2}\sigma_{ij}^{2} ~,\\
b_{mj}=\gamma_{mj}+\sqrt{\delta_{mj}}\epsilon_{mj}\quad\textrm{with}\quad \epsilon_{mj} \sim \mathcal{N}(0,1) ~,
\end{gathered}
$$

where $$\boldsymbol{\epsilon}$$ is a matrix of the same size as $$\mathbf{B}$$. The difference between the two reparametrization tricks is visualized in the following.

<div>
  <img src="/assets/imgs/BNNs_with_VI_files/rt-1.png" width="200"/>
  <img src="/assets/imgs/BNNs_with_VI_files/lrt-1.png" width="200"/>
</div>

### Implementation

#### General VI Module

To later be able to extend the concept to the local reparametrization trick, we start by implementing a general VI module, which lies at the heart of all further layers.


```python
from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

class VIModule(nn.Module):

    def __init__(self,
                 layer_fct: Callable,
                 weight_size: tuple,
                 bias_size: tuple = None,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'reverse'):

        super(VIModule, self).__init__()

        # function for forward pass e.g. F.linear, F.conv2d
        self.layer_fct = layer_fct

        # fall back to default vals
        if prior is None:
            prior = {'mu': 0, 'sigma': 0.1}

        if posteriors is None:
            posteriors = {
                'mu': (0, 0.1),
                'rho': (-3., 0.1)
            }

        # if prior is ScaleMixture we must use MC integration for KL div.
        # otherwise we can compute KL div. analitically
        if 'pi' in list(prior.keys()):
            self._kl_divergence = mc_kl_divergence
            self.prior = MixtureNormal(prior['mu'], prior['sigma'], prior['pi'])
        else:
            self._kl_divergence = kl_divergence
            self.prior = Normal(prior['mu'], prior['sigma'])

        # either 'forward' or 'reverse'
        self.kl_type = kl_type

        # save parameters for resetting
        self.posterior_mu_initial = posteriors['mu']
        self.posterior_rho_initial = posteriors['rho']

        # initialize weights and biases
        self.W_mu = nn.Parameter(torch.empty(weight_size))
        self.W_rho = nn.Parameter(torch.empty(weight_size))
        if bias_size is not None:
            self.bias_mu = nn.Parameter(torch.empty(bias_size))
            self.bias_rho = nn.Parameter(torch.empty(bias_size))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # reset
        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.bias_mu is not None:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    @property
    def kl(self):
        # compute KL div. by instantiating the weights as Normal distribution
        _kl = self.kl_divergence(self.prior, Normal(self.W_mu.cpu(), F.softplus(self.W_rho).cpu())).sum()
        if self.bias_mu is not None:
            _kl += self.kl_divergence(self.prior, Normal(self.bias_mu.cpu(), F.softplus(self.bias_rho).cpu())).sum()
        return _kl

    def kl_divergence(self, p: Distribution, q: Distribution,
                      kl_type: str = 'reverse') -> torch.Tensor:
        # either reverse or forward KL div.
        if kl_type == 'reverse':
            return self._kl_divergence(p, q)
        else:
            return self._kl_divergence(q, p)

    @staticmethod
    def rsample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # reparametrization trick
        eps = torch.empty(mu.size()).normal_(0, 1).to(mu.device)
        return mu + eps * sigma
```

In this post we will be using linear layers only, but the module can easily be extended to a convolutional layer by changing the layer function. The VI module can be given a prior as well as a posterior distribution. Dependend on whether a mixture of Gaussians shall be used as prior, the KL divergence is either computed analytically with the version provided by PyTorch or our MC KL divergence is used. Further we can specify, whether the forward or reverse KL divergence shall be computed. At last, it implements a function for performing reparametrization.

#### (Local) Reparametrization Trick Layer

Now we extend our VI module with the reparametrization trick and its local counterpart.


```python
class RTLayer(VIModule):

    def __init__(self,
                 layer_fct: Callable,
                 weight_size: tuple,
                 bias_size: tuple = None,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'forward',
                 **kwargs):

        super(RTLayer, self).__init__(layer_fct=layer_fct,
                                      weight_size=weight_size,
                                      bias_size=bias_size,
                                      prior=prior,
                                      posteriors=posteriors,
                                      kl_type=kl_type)
        # these will be used for an easy extension to a convolutional layer
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sample each weight with reparametrization trick
        weight = self.rsample(self.W_mu, F.softplus(self.W_rho))
        if self.bias_mu is not None:
            bias = self.rsample(self.bias_mu, F.softplus(self.bias_rho))
        else:
            bias = None
        # use this weight for forward pass
        return self.layer_fct(x, weight, bias, **self.kwargs)


class LRTLayer(VIModule):

    def __init__(self,
                 layer_fct: Callable,
                 weight_size: tuple,
                 bias_size: tuple = None,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'forward',
                 **kwargs):

        super(LRTLayer, self).__init__(layer_fct=layer_fct,
                                       weight_size=weight_size,
                                       bias_size=bias_size,
                                       prior=prior,
                                       posteriors=posteriors,
                                       kl_type=kl_type)
        # these will be used for an easy extension to a convolutional layer
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # first conduct forward pass for mean and variance
        act_mu = self.layer_fct(x, self.W_mu, self.bias_mu, **self.kwargs)
        self.W_sigma = F.softplus(self.W_rho)
        if self.bias_mu is not None:
            bias_var = F.softplus(self.bias_rho) ** 2
        else:
            bias_var = None
        act_std = torch.sqrt(1e-16 + self.layer_fct(x**2, self.W_sigma**2, bias_var, **self.kwargs))
        # sample from activation
        return self.rsample(act_mu, act_std)
```

The implementation for both the "normal" reparametrization layer (`RTLayer`) and the local reparametrization layer (`LRTLayer`) both look very similar. The only difference can be spotted in the forward function. As described earlier the RTLayer first samples the weight vector and then performs the forward pass. In contrast to that the LRTLayer first performs the forward pass for mean and variance and afterwards uses the reparametrization trick to sample from these activations.

#### Linear Layer

The extensions to real layers, linear and convolutional, is trivial now. The example for a linear layer can be found underneath, the implementation for the convolutional layers can be found at the end of the post.


```python
class LinearRT(RTLayer):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'forward'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearRT, self).__init__(layer_fct=F.linear,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type)

class LinearLRT(LRTLayer):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'forward'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearLRT, self).__init__(layer_fct=F.linear,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type)
```

#### Loss Function

Before we can start using these layers in a neural network, we must implement our loss function, the ELBO. As a loss function we use the full negative log likelihood. This has the advantage of being able to estimate the full uncertainty. We will not dive deeper into the derivations, this is done in another <a href="">post</a>. As a quick overview, uncertainty can be divided into uncertainty or noise inherent in the data, called aleatoric uncertainty, and uncertainty inherent in our model, termed epistemic uncertainty. The aleatoric uncertainty is captured implictly during training with our loss function and can also be estimated without employing Bayesian techniques. To capture epistemic uncertainty on the other hand we must impose distributions onto our parameters and follow the Bayesian approach.

To capture the aleatoric uncertainty our model $$\mathbf{f}_{\boldsymbol{\theta}}$$ gains a new head

$$
\mathbf{f}_{\boldsymbol{\theta}}(\mathbf{x})=\left[ \hat{\mathbf{y}},\hat{\boldsymbol{\sigma}}^2 \right]
$$

and our loss function is then given by

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{D}\sum_i \frac{1}{2}\hat{\sigma}_i^2 \left( y_i - \hat{y}_i \right)^2 + \frac{1}{2}\log\hat{\sigma}_i^2 ~.
$$

Using this loss function we can implicitly learn the aleatoric uncertainty in our data (<a href="https://arxiv.org/abs/1703.04977">Kendall and Gal 2017</a>). For numerical stability in practice we let our model ouput the (negative) log variance as this allows for negative values as well:

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{D}\sum_i\frac{1}{2}\exp \left( -\log \hat{\sigma}_i^2 \right) \left( y_i - \hat{y}_i \right)^2 - \frac{1}{2}\log\hat{\sigma}_i^2 ~.
$$

In Regression $$D$$ is set to the number of samples in the mini-batch, while in classification $$D=1$$. To also account for epistemic uncertainty we need to compute the variance of the output using Monte Carlo integration

$$
\mathrm{Var}\left[\mathbf{y}\right]=\underbrace{\frac{1}{T}\sum_{t=1}^{T}\hat{\mathbf{y}}_t^2-\left(\frac{1}{T}\sum_{t=1}^{T}\hat{\mathbf{y}}_t \right)^2}_{\textrm{epistemic}}+\underbrace{\boldsymbol{\sigma}}_{\textrm{aleatoric}} ~,
$$

where $$\boldsymbol{\sigma}$$ denotes the aleatoric part estimated with the full negative log likelihood.


```python
def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                 reduction: str = 'mean') -> torch.Tensor:
    loss = torch.exp(-logvar) * torch.pow(target - mu, 2) + logvar
    return loss.mean() if reduction=='mean' else loss.sum()

class ELBO(nn.Module):
    def __init__(self, train_size: int = 1, train_type: str = 'regression'):        
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.reduction = 'mean' if train_type == 'regression' else 'sum'

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, kl: torch.Tensor,
                beta: float = 1.) -> torch.Tensor:
        return gaussian_nll(inputs[:,0], inputs[:,1], target[:,0], self.reduction) * self.train_size + beta * kl

def calc_uncert(preds: [torch.Tensor], reduction: str = 'mean') -> (torch.Tensor, torch.Tensor, torch.Tensor):
    preds = torch.cat(preds, dim=0)
    epi = torch.var(preds[:,:,0], dim=0)
    ale = torch.mean(preds[:,:,1], dim=0).exp()
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean(), epi.mean(), uncert.mean()
    else:
        return ale, epi, uncert
```

#### KL Divergence Reweighting

The attentive reader may have noticed that we have reweighted both our likelihood as well as our KL divergence in the above code for the ELBO. A problem in training Bayesian neural networks with VI arises when we have a discrepancy between number of model parameters and data set size. Most often our number of parameters exceeds the number of training points leading to overfitting when performing maximum likelihood estimation. When utilizing the ELBO this means the magnitude of the KL divergence term, the regularizer, exceeds the likelihood cost and the training focusses on reducing the complexity instead of the likelihood. Thus, the KL divergence term must be reweighted by a factor $$\beta$$:

$$
\textrm{ELBO}(q(\boldsymbol{\theta})) = \mathbb{E}_{\boldsymbol{\theta}\sim q} \log p(\mathcal{D}|\boldsymbol{\theta}) - \beta \mathrm{KL}(q(\boldsymbol{\theta})||p(\boldsymbol{\theta})) ~.
$$

A good value for $$\beta$$ leads to an initial magnitude of the complexity cost comparable to the magnitude of the likelihood term. But considering only the likelihood disregards the number of model parameters, which highly influences the magnitude of the KL divergence. Following from this, the KL divergence must be scaled by the number of parameters to ensure its magnitude is only influenced by the approximation between posterior and prior. Reducing the number of parameters would otherwise decrease the complexity cost without increasing the model fit. At best, the likelihood is scaled by the number of data points in the data set to balance out both terms.

Since the use for the ELBO is motivated by enabling mini-batched training, the literature provides different $$\beta$$ for training with mini-batches. The standard scaling factor was introduced by <a href="https://www.cs.toronto.edu/~graves/nips_2011.pdf">Graves (2011)</a>, which sets $$\beta=\frac{1}{M}$$, where $$M$$ denotes the number of mini-batches. A more sophisticated version is provided by <a href="https://arxiv.org/abs/1505.05424">Blundell et al. (2015)</a>:

$$\beta_i = \frac{2^{M-i}}{2^M-1} ~, \quad i \in \{1,...,M\} ~,$$

where $$i$$ denotes the number of the current batch number. This condition ensures that $$\beta$$ is not uniform across mini-batches but still sophisticates $$\sum_{i=1}^M \beta_i=1$$. While assuming higher values for $$\beta$$ at the beginning of training, the importance of the complexity costrapidly declines. By utilizing this approach the prior gets more influential in the beginning,before the training focuses on the likelihood, when more data is observed.


```python
from typing import Union

def get_beta(beta_type: Union[float, str] = 1., batch_idx: int = 1, m: int = 1,
             epoch: int = 1, num_epochs: int = 1) -> float:
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = beta_type
    return beta
```

Before we conduct experiments with the above described VI framework, we will revisit another very popular technique for approximate variational inference in neural networks that gains its popularity from its simplicity and fewer computational requirements.

## Monte Carlo Dropout

Coming from the rather intuitive way of modelling each weight with a Gaussian distribution to incorporate uncertainty into the model parameters, Monte Carlo (MC) dropout is less instinctive but bears some strong advantages. <a href="https://arxiv.org/abs/1506.02142">Gal and Ghahramani (2016)</a> proposed dropout as approximate Bayesian inference. Their idea is basically simple, instead of just applying dropout during training to prevent overfitting dropout is also applied during testing. This makes the output of the model a random variable to and we are able to quantify uncertainty is described earlier.

The derivation relies on variational inference and is based on some heavy assumptions about the prior. For the full proof the interested reader is referred to <a href="http://proceedings.mlr.press/v48/gal16-supp.pdf">Gal and Ghahramani (2015)</a>.

An MC dropout layer can easily be implemented in PyTorch.


```python
class MCDropout(nn.Module):
    def __init__(self, p: float):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True)
```

### Experiments

As we have now revisited the most popular approxmation techniques with variational inference, we will now take them to use for the problem of non-linear regression. So, we first must generate our training data:

$$
y = 10 \sin(2\pi x) + \epsilon \quad \textrm{with} \quad \epsilon \sim \mathcal{N}(0,2) ~.
$$


```python
import numpy as np

def f(x: torch.Tensor, sigma: float) -> torch.Tensor:
    epsilon = torch.randn(*x.shape) * sigma
    return 10 * torch.sin(2 * np.pi * (x)) + epsilon

train_size = 92
sigma = 2.0

x = torch.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(x, sigma=sigma)
y_true = f(x, sigma=0.0)

plt.scatter(x, y, color='black', label='Training data')
plt.plot(x, y_true, label='True', color='r', linestyle='--')
plt.legend()
plt.show()
```



![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_22_0.png)



Next we will generate helper functions that simplify training lateron. Also, we defined our function that plots the uncertainty and the mean prediction of our models.


```python
from scipy.ndimage import gaussian_filter1d
from torch.optim import Optimizer, Adam

def train(model: nn.Module, loss_fct: Callable, x: torch.Tensor, y: torch.Tensor,
          batch_size: int = 1, beta_type: Union[float, str] = 1., optim: Optimizer = Adam,
          weight_decay: float = 0., lr: float = 1e-3, num_epochs: int = 1000) -> (list):

    losses, mses, kls, grads = [], [], [], []

    optim = optim(model.parameters(), lr=lr, weight_decay=weight_decay)
    m = int(len(x)/batch_size)
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        idx = torch.randperm(len(x))
        total_loss, total_kl = 0, 0
        for batch_idx in range(1, m+1):
            optim.zero_grad()

            out, kl = model(x[idx[(batch_idx-1) * batch_size : batch_idx * batch_size]])

            beta = get_beta(beta_type=beta_type, batch_idx=batch_idx, m=m, epoch=epoch, num_epochs=num_epochs)
            loss = loss_fct(out, y[idx[(batch_idx-1) * batch_size : batch_idx * batch_size]], kl, beta)

            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_kl += beta * kl.item()

            mse = F.mse_loss(out, y[idx[(batch_idx-1) * batch_size : batch_idx * batch_size]]).detach()
            mses.append(mse.item())

        losses.append(total_loss)
        kls.append(total_kl)
        grads.append(model.gradients())

        pbar.set_description('loss: %.6f' % loss.item())

    return losses, mses, kls, grads


def pred(model: nn.Module, x: torch.Tensor, mc_samples: int = 100) -> (torch.Tensor):

    y_preds = []

    with torch.no_grad():
        for _ in tqdm(range(mc_samples)):
            y_pred, _ = model(x)
            y_preds.append(y_pred.unsqueeze(0))

    y_mean = torch.cat(y_preds, dim=0)[:,:,0].mean(dim=0)
    if y_preds[0].shape[-1] > 1:
        ale, epi, uncert = calc_uncert(y_preds, reduction=None)
    else:
        ale, epi, uncert = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])

    return y_mean, ale, epi, uncert


def plot_uncert(x_test: torch.Tensor, y_pred_mean: torch.Tensor, x_train: torch.Tensor, y_train: torch.Tensor,
                ale: torch.Tensor, epi: torch.Tensor, uncert: torch.Tensor):
    fig, ax = plt.subplots()

    ale, epi, uncert = torch.sqrt(ale), torch.sqrt(epi), torch.sqrt(uncert)
    ax.plot(x_test, y_pred_mean, color='#D1895C', label='Predictive mean');
    ax.scatter(x_train, y_train, color='black', label='Training data')
    ax.fill_between(x_test.flatten(),
                    gaussian_filter1d(y_pred_mean + 2 * (ale + epi), sigma=5),
                    gaussian_filter1d(y_pred_mean - 2 * (ale + epi), sigma=5),
                    color='#6C85B6',
                    alpha=0.3, label='Aleatoric uncertainty')
    ax.fill_between(x_test.flatten(),
                    gaussian_filter1d(y_pred_mean + 2 * epi, sigma=5),
                    gaussian_filter1d(y_pred_mean - 2 * epi, sigma=5),
                    color='#6C85B6',
                    alpha=0.5, label='Epistemic uncertainty')
    ax.set_xlabel(r'$x$', fontsize=17)
    ax.set_ylabel(r'$y$', fontsize=17)
    ax.legend()
    return fig
```

#### Hyperparameters


```python
LR = 0.08
BATCH_SIZE = 8
NUM_EPOCHS = 200
```

### Frequentist Approach

Before we conduct experiments with variational inference, we will first take a look at the frequentist approach.


```python
class Model(nn.Module):

    def __init__(self, in_features: int = 1, out_features: int = 1):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(in_features, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, out_features)

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # as the training function expects the model to return the kl, we return 0 here
        return x, torch.tensor([0])

    def gradients(self) -> float:
        return torch.cat([param.grad.flatten() for param in self.parameters()]).sum().item()
```


```python
model_freq = Model()
loss_fct = lambda out, y, kl, beta : F.mse_loss(out, y)
_, _, _, _ = train(model=model_freq, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                beta_type=0., lr=LR, num_epochs=NUM_EPOCHS)
```

    loss: 3.760674: 100%|██████████| 200/200 [00:02<00:00, 75.08it/s]



```python
x_test = torch.linspace(-1.5, 1.5, 500).reshape(-1, 1)

y_pred_mean_freq, ale_freq, epi_freq, uncert_freq = pred(model=model_freq, x=x_test)
fig = plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_freq, x_train=x, y_train=y,
                  ale=ale_freq, epi=epi_freq, uncert=uncert_freq)
```

    100%|██████████| 100/100 [00:00<00:00, 4270.19it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_30_1.png)



As can be seen, the output of the frequentist model only has one defined value. Even in regions lacking training data the model confidently outputs one value as the only true one. Moving forward to a Bayesian approach, we will first conduct maximum a posterior inference utilizing the full negativ elog likelihood as loss function and by this equipping our model with the ability to quantify aleatoric uncertainty in the data.


```python
model_map = Model(out_features=2)
loss_fct = lambda out, y, kl, beta : gaussian_nll(out[:,0], out[:,1], y[:,0], 'mean')
_, _, _, _ = train(model=model_map, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                   beta_type=0., lr=LR, num_epochs=NUM_EPOCHS)
```

    loss: 2.771659: 100%|██████████| 200/200 [00:03<00:00, 64.52it/s]



```python
y_pred_mean_map, ale_map, epi_map, uncert_map = pred(model=model_map, x=x_test)
fig = plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_map, x_train=x, y_train=y,
                  ale=ale_map, epi=epi_map, uncert=uncert_map)
```

    100%|██████████| 100/100 [00:00<00:00, 4455.30it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_33_1.png)



With this approach we are able to incorporate quantitative uncertainty estimates into the model's prediction. The farther we get from the training data, the more the uncertainty in the model's output increases. Still, this is only part of the uncertainty, we do still not consider the uncertainty inherent in the model. Thus, we will now move towards the fully Bayesian approach with variational inference.

### Going Bayesian with Monte Carlo Dropout

The first experiments will be conducted with MC dropout to show its simplicity. This approach only needs a very simple adjustment, we only must use the MCDropout layer from above after every forward pass.


```python
class ModelDropout(nn.Module):

    def __init__(self, in_features: int = 1, out_features: int = 2, p: float = 0.5):
        super(ModelDropout, self).__init__()

        self.layer1 = nn.Linear(in_features, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, out_features)

        self.mc_dropout = MCDropout(p=p)

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        x = F.relu(self.mc_dropout(self.layer1(x)))
        x = F.relu(self.mc_dropout(self.layer2(x)))
        x = self.mc_dropout(self.layer3(x))
        # as the training function expects the model to return the kl, we return 0 here
        return x, torch.tensor([0])

    def gradients(self) -> float:
        return torch.cat([param.grad.flatten() for param in self.parameters()]).sum().item()
```

It is important to mention that we must always set a weight decay in the optimizer, since that determines the shape of our prior (together with the dropout rate).


```python
p = 0.01
weight_decay = 0.001

model_dropout = ModelDropout(p=p)
loss_fct = lambda out, y, kl, beta : gaussian_nll(out[:,0], out[:,1], y[:,0], 'mean')
_, _, _, _ = train(model=model_dropout, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                   beta_type=0., lr=LR, num_epochs=NUM_EPOCHS, weight_decay=weight_decay)
```

    loss: 3.081201: 100%|██████████| 200/200 [00:03<00:00, 56.73it/s]



```python
y_pred_mean_dropout, ale_dropout, epi_dropout, uncert_dropout = pred(model=model_dropout, x=x_test)
fig = plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_dropout, x_train=x, y_train=y,
                  ale=ale_dropout, epi=epi_dropout, uncert=uncert_dropout)
plt.tight_layout()
#plt.savefig('dropout_uncert.pdf', bbox_inches='tight')
```

    100%|██████████| 100/100 [00:00<00:00, 1353.97it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_38_1.png)



Dropout makes it possible to quantify both, aleatoric and epistemic, uncertainty.

### Bayes by Backprop

We will now use the VI layer we have defined earlier. Since we are modeling out prior and posterior explicitly (at the cost of doubling the number of weights), we have more freedom in designing the training procedure. The only difference to the models above is that we now have a KL divergence that must be computed for each forward pass


```python
class ModelVI(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 2,
                 prior: dict = None, posteriors: dict = None, reparam: str = 'lrt'):
        super(ModelVI, self).__init__()
        LinearVI = LinearLRT if reparam == 'lrt' else LinearRT
        self.layer1 = LinearVI(in_features, 20, prior=prior, posteriors=posteriors)
        self.layer2 = LinearVI(20, 20, prior=prior, posteriors=posteriors)
        self.layer3 = LinearVI(20, out_features, prior=prior, posteriors=posteriors)

    def forward(self, x: torch.Tensor) -> (torch.Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x, self.kl_div()

    def kl_div(self) -> torch.Tensor:
        kl = 0
        for m in self.children():
            kl += m.kl
        return kl

    def gradients(self) -> float:
        return torch.cat([param.grad.flatten() for param in self.parameters()]).sum().item()
```

In our first training procedure we will use a $$\beta=10^{-2}$$ for both the "standard" and local reparametrization trick.


```python
prior = {'mu': 0, 'sigma': 0.01}

posteriors = {
    'mu': (0, 0.1),
    'rho': (-3., 0.1)
}

elbo = ELBO(train_size=len(x))
loss_fct = lambda y, out, kl, beta: elbo(y, out, kl, beta)

kls, preds, mses = {}, {}, {}
```

#### "Standard" Reparametrization Trick


```python
model_vi_rt = ModelVI(prior=prior, posteriors=posteriors, reparam='rt')

no_params = lambda model: len(torch.cat([p.flatten() for p in model.parameters()]))
beta_type = 1/no_params(model_vi_rt)

losses_rt, mses_rt, kls_rt, grads_rt = train(model=model_vi_rt, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                                             beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
```

    loss: 341.115906: 100%|██████████| 200/200 [00:08<00:00, 22.25it/s]



```python
y_pred_mean_rt, ale_rt, epi_rt, uncert_rt = pred(model=model_vi_rt, x=x_test)
fig = plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_rt, x_train=x, y_train=y,
                  ale=ale_rt, epi=epi_rt, uncert=uncert_rt)
```

    100%|██████████| 100/100 [00:00<00:00, 945.32it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_45_1.png)



#### Local Reparametrization Trick


```python
model_vi_lrt = ModelVI(prior=prior, posteriors=posteriors)

beta_type = 1/no_params(model_vi_lrt)

losses_lrt, mses_lrt, kls_lrt, grads_lrt = train(model=model_vi_lrt, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                                                 beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)

kls[r'\#weights'] = kls_lrt
mses[r'\#weights'] = mses_lrt
```

    loss: 286.557373: 100%|██████████| 200/200 [00:10<00:00, 19.43it/s]



```python
y_pred_mean_lrt, ale_lrt, epi_lrt, uncert_lrt = pred(model=model_vi_lrt, x=x_test)
fig = plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_lrt, x_train=x, y_train=y,
                  ale=ale_lrt, epi=epi_lrt, uncert=uncert_lrt)
plt.tight_layout()
#plt.savefig('ffg_uncert.pdf', bbox_inches='tight')
preds[r'\#weights'] = y_pred_mean_lrt
```

    100%|██████████| 100/100 [00:00<00:00, 623.07it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_48_1.png)



#### Comparing Gradients

With more training samples the figure underneath would get more meaningful, but also in this example it can bessen that the average gradients after each forward pass are lower for the LRT meaning our model does not oscillate that much around a lcoal optimum.


```python
plt.plot(range(len(grads_rt)), gaussian_filter1d(grads_rt, sigma=1), label=r'RT')
plt.plot(range(len(grads_lrt)), gaussian_filter1d(grads_lrt, sigma=1), label=r'LRT')
plt.legend()
plt.xlabel(r'iteration', fontsize=17)
plt.ylabel(r'grad', fontsize=17)
plt.show()
```



![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_50_0.png)



### Scale Mixture Prior

<a href="https://arxiv.org/abs/1505.05424">Blundell et al. (2015)</a> have shown that using a scale mixture prior (mixture of two Gaussians) can be benefitial for training, since then the model has the ability to have weights with a very small variance but can also match for uncertainty estimation the high variance part of the prior.


```python
prior = {'mu': [0., 0.], 'sigma': [0.001, 100.], 'pi': [0.5, 0.5]}

posteriors = {
    'mu': (0, 0.1),
    'rho': (-3., 0.1)
}

model_vi_smp = ModelVI(prior=prior, posteriors=posteriors)

beta_type = 1/no_params(model_vi_smp)

losses_smp, mses_smp, kls_smp, grads_smp = train(model=model_vi_smp, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                                       beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
```

    loss: 192.050583: 100%|██████████| 200/200 [00:17<00:00, 11.24it/s]



```python
y_pred_mean_smp, ale_smp, epi_smp, uncert_smp = pred(model=model_vi_smp, x=x_test)
plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_smp, x_train=x, y_train=y,
            ale=ale_smp, epi=epi_smp, uncert=uncert_smp)
```

    100%|██████████| 100/100 [00:00<00:00, 230.19it/s]






![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_53_1.png)






![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_53_2.png)



### Different Betas

Since the selection of $$\beta$$ highly influences our training procedure, we will conduct experiments using different $$\beta$$.

#### Graves


```python
prior = {'mu': 0, 'sigma': 0.01}
beta_type = "Standard"
model_vi_lrt_standard = ModelVI(prior=prior, posteriors=posteriors)
_, mses_lrt_standard, kls_lrt_standard, _ = train(model=model_vi_lrt_standard, loss_fct=loss_fct, x=x, y=y,
                                                  batch_size=BATCH_SIZE, beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
kls[beta_type] = kls_lrt_standard
mses[beta_type] = mses_lrt_standard
```

    loss: 406.891388: 100%|██████████| 200/200 [00:10<00:00, 18.78it/s]



```python
y_pred_mean_lrt, ale_lrt, epi_lrt, uncert_lrt = pred(model=model_vi_lrt_standard, x=x_test)
plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_lrt, x_train=x, y_train=y,
            ale=ale_lrt, epi=epi_lrt, uncert=uncert_lrt)
preds[beta_type] = y_pred_mean_lrt
```

    100%|██████████| 100/100 [00:00<00:00, 555.56it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_56_1.png)



#### Blundell


```python
beta_type = "Blundell"
model_vi_lrt_blundell = ModelVI(prior=prior, posteriors=posteriors)
_, mses_lrt_blundell, kls_lrt_blundell, _ = train(model=model_vi_lrt_blundell, loss_fct=loss_fct, x=x, y=y,
                                                 batch_size=BATCH_SIZE, beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
kls[beta_type] = kls_lrt_blundell
mses[beta_type] = mses_lrt_blundell
```

    loss: 323.418182: 100%|██████████| 200/200 [00:10<00:00, 18.80it/s]



```python
y_pred_mean_lrt, ale_lrt, epi_lrt, uncert_lrt = pred(model=model_vi_lrt_blundell, x=x_test)
plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_lrt, x_train=x, y_train=y,
            ale=ale_lrt, epi=epi_lrt, uncert=uncert_lrt)
preds[beta_type] = y_pred_mean_lrt
```

    100%|██████████| 100/100 [00:00<00:00, 457.78it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_59_1.png)



#### No Reweighting ($$\beta=1$$)


```python
beta_type = 1.
model_vi_lrt_1 = ModelVI(prior=prior, posteriors=posteriors)
_, mses_lrt_1, kls_lrt_1, _ = train(model=model_vi_lrt_1, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                                    beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
kls[beta_type] = kls_lrt_1
mses[beta_type] = mses_lrt_1
```

    loss: 527.517944: 100%|██████████| 200/200 [00:10<00:00, 19.60it/s]



```python
y_pred_mean_lrt, ale_lrt, epi_lrt, uncert_lrt = pred(model=model_vi_lrt_1, x=x_test)
plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_lrt, x_train=x, y_train=y,
            ale=ale_lrt, epi=epi_lrt, uncert=uncert_lrt)
preds[beta_type] = y_pred_mean_lrt
```

    100%|██████████| 100/100 [00:00<00:00, 653.01it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_62_1.png)



#### No KL divergence ($$\beta=0$$)


```python
beta_type = 0.
model_vi_lrt_0 = ModelVI(prior=prior, posteriors=posteriors)
_, mses_lrt_0, kls_lrt_0, _ = train(model=model_vi_lrt_0, loss_fct=loss_fct, x=x, y=y, batch_size=BATCH_SIZE,
                                    beta_type=beta_type, lr=LR, num_epochs=NUM_EPOCHS)
kls[beta_type] = kls_lrt_0
mses[beta_type] = mses_lrt_0
```

    loss: 183.360672: 100%|██████████| 200/200 [00:10<00:00, 19.51it/s]



```python
y_pred_mean_lrt, ale_lrt, epi_lrt, uncert_lrt = pred(model=model_vi_lrt_0, x=x_test)
plot_uncert(x_test=x_test, y_pred_mean=y_pred_mean_lrt, x_train=x, y_train=y,
            ale=ale_lrt, epi=epi_lrt, uncert=uncert_lrt)
preds[beta_type] = y_pred_mean_lrt
```

    100%|██████████| 100/100 [00:00<00:00, 624.75it/s]




![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_65_1.png)




```python
colors = {'\#weights': 0, 'Standard': 1, 'Blundell': 2, 1.0: 3, 0.0: 4}
for beta, kl in kls.items():
    plt.plot(range(len(kl)), kl, label=beta, color=sns.color_palette()[colors[beta]])
plt.legend(loc='upper right')
plt.ylim([-20, 1695])
plt.xlabel(r'iteration', fontsize=17)
plt.ylabel(r'KL$q((\bm{\theta})||p(\bm{\theta}))$', fontsize=17)
plt.tight_layout()
plt.savefig('diff_betas_kl.pdf', bbox_inches='tight')
```



![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_66_0.png)




```python
for beta, pred in preds.items():
    plt.plot(x_test, pred, label=beta, color=sns.color_palette()[colors[beta]])
plt.scatter(x, y, color='black', label='training data')
plt.legend(loc='upper right')
plt.xlabel(r'x', fontsize=17)
plt.ylabel(r'y', fontsize=17)
plt.tight_layout()
plt.savefig('diff_betas_pred.pdf', bbox_inches='tight')
```



![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_67_0.png)




```python
for beta, mse in mses.items():
    plt.plot(range(len(mse)), gaussian_filter1d(mse, sigma=10), label=beta, color=sns.color_palette()[colors[beta]])
plt.legend(loc='upper right')
plt.xlabel(r'iteration', fontsize=17)
plt.ylabel(r'MSE$(y;\hat{y})$', fontsize=17)
plt.tight_layout()
plt.savefig('diff_betas_nll.pdf', bbox_inches='tight')
```



![png](/assets/imgs/BNNs_with_VI_files/BNNs_with_VI_68_0.png)



## Appendix


```python
from torch.nn.functional import conv2d
from torch.nn.modules.utils import _pair

class Conv2dRT(RTLayer):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 bias: bool = True,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        weight_size = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        bias_size = (out_channels) if bias else None

        super(Conv2dRT, self).__init__(layer_fct=conv2d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)


class Conv2dLRT(LRTLayer):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 bias: bool = True,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'reverse'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        weight_size = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])

        bias_size = (out_channels) if bias else None

        super(Conv2dLRT, self).__init__(layer_fct=conv2d,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups)
```


```python

```
