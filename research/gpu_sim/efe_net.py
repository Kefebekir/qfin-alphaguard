"""Efe's network for the GPU gym (pre-registration G1).

One hidden layer: the ten observations go through `hidden` tanh units to one output, an
exposure between -2 and 2. All the network's numbers live in one row of `params`, so that
evolution strategies can try hundreds of versions of it at once. A network that does not
see the rule gets 0 in place of the rule's position (observation 8).
"""

import torch

FEATURES = 10
RULE = 8  # the observation that holds the rule's position


def make(hidden=16, see_rule=True):
    """The policy and how many numbers it has."""
    size = FEATURES * hidden + hidden + hidden + 1

    def policy(obs, params, market):
        if not see_rule:
            obs = obs.clone()
            obs[..., RULE] = 0.0
        v = params.shape[0]
        w1 = params[:, : FEATURES * hidden].view(v, FEATURES, hidden)
        b1 = params[:, FEATURES * hidden : FEATURES * hidden + hidden]
        w2 = params[:, FEATURES * hidden + hidden : FEATURES * hidden + 2 * hidden]
        b2 = params[:, -1:]
        h = torch.tanh(torch.einsum("vnf,vfh->vnh", obs, w1) + b1[:, None, :])
        return 2 * torch.tanh(torch.einsum("vnh,vh->vn", h, w2) + b2)

    return policy, size
