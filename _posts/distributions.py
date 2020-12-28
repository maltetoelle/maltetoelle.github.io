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
