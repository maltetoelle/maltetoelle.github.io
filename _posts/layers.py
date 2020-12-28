from typing import Callable, Union

import torch
from torch.nn import Parameter, Module
from torch.nn.functional import conv2d, linear, dropout, dropout2d, softplus
from torch.nn.modules.utils import _pair
from torch.distributions.kl import kl_divergence
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal

from distributions import mc_kl_divergence, MixtureNormal


class MCDropout(Module):
    def __init__(self, p: float, dim: str = '1d'):
        super(MCDropout, self).__init__()
        self.p = p
        self._dropout = dropout2d if dim == '2d' else dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(x, self.p, training=True)


class VIModule(Module):

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
        self.W_mu = Parameter(torch.empty(weight_size))
        self.W_rho = Parameter(torch.empty(weight_size))
        if bias_size is not None:
            self.bias_mu = Parameter(torch.empty(bias_size))
            self.bias_rho = Parameter(torch.empty(bias_size))
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
        _kl = self.kl_divergence(Normal(self.W_mu.cpu(), softplus(self.W_rho).cpu()), self.prior, self.kl_type).sum()
        if self.bias_mu is not None:
            _kl += self.kl_divergence(Normal(self.bias_mu.cpu(), softplus(self.bias_rho).cpu()), self.prior, self.kl_type).sum()
        return _kl

    def kl_divergence(self, p: Distribution, q: Distribution, kl_type: str = 'reverse') -> torch.Tensor:
        # either reverse or forward KL div.
        if kl_type == 'reverse':
            return self._kl_divergence(q, p)
        else:
            return self._kl_divergence(p, q)

    @staticmethod
    def rsample(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # reparametrization trick
        eps = torch.empty(mu.size()).normal_(0, 1).to(mu.device)
        return mu + eps * sigma

class RTLayer(VIModule):

    def __init__(self,
                 layer_fct: Callable,
                 weight_size: tuple,
                 bias_size: tuple = None,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'reverse',
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
        weight = self.rsample(self.W_mu, softplus(self.W_rho))
        if self.bias_mu is not None:
            bias = self.rsample(self.bias_mu, softplus(self.bias_rho))
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
                 kl_type: str = 'reverse',
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
        self.W_sigma = softplus(self.W_rho)
        if self.bias_mu is not None:
            bias_var = softplus(self.bias_rho) ** 2
        else:
            bias_var = None
        act_std = torch.sqrt(1e-16 + self.layer_fct(x**2, self.W_sigma**2, bias_var, **self.kwargs))
        # sample from activation
        return self.rsample(act_mu, act_std)


class LinearRT(RTLayer):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 prior: dict = None,
                 posteriors: dict = None,
                 kl_type: str = 'reverse'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearRT, self).__init__(layer_fct=linear,
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
                 kl_type: str = 'reverse'):

        self.in_features = in_features
        self.out_featurs = out_features

        weight_size = (out_features, in_features)

        bias_size = (out_features) if bias else None

        super(LinearLRT, self).__init__(layer_fct=linear,
                                        weight_size=weight_size,
                                        bias_size=bias_size,
                                        prior=prior,
                                        posteriors=posteriors,
                                        kl_type=kl_type)


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
