import numpy as np
import torch

from ....modules import sparse as sp
from ....pipelines import samplers
from ....utils.general_utils import dict_foreach


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"

        if self.p_uncond > 0:
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                if isinstance(cond, sp.SparseTensor):
                    return cond.shape[0]
                if isinstance(cond, list):
                    return len(cond)
                raise ValueError(f"Unsupported type of cond: {type(cond)}")

            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            batch_size = get_batch_size(ref_cond)

            def select(pos_cond, neg_cond, mask):
                if isinstance(pos_cond, torch.Tensor):
                    mask_tensor = torch.tensor(mask, device=pos_cond.device).reshape(-1, *[1] * (pos_cond.ndim - 1))
                    return torch.where(mask_tensor, neg_cond, pos_cond)
                if isinstance(pos_cond, sp.SparseTensor):
                    pos_unbind = sp.sparse_unbind(pos_cond, dim=0)
                    neg_unbind = sp.sparse_unbind(neg_cond, dim=0)
                    chosen = [nu if m else pu for pu, nu, m in zip(pos_unbind, neg_unbind, mask)]
                    return sp.sparse_cat(chosen, dim=0)
                if isinstance(pos_cond, list):
                    return [nc if m else c for c, nc, m in zip(pos_cond, neg_cond, mask)]
                raise ValueError(f"Unsupported type of cond: {type(pos_cond)}")

            mask = (np.random.rand(batch_size) < self.p_uncond).tolist()
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {"cond": cond, "neg_cond": neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        return samplers.FlowEulerCfgSampler(self.sigma_min)
