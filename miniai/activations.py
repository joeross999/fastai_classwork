# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/10_activations.ipynb.

# %% ../notebooks/10_activations.ipynb 1
from __future__ import annotations
import random,math,torch,numpy as np,matplotlib.pyplot as plt
import fastcore.all as fc
from functools import partial

from .datasets import *
from .learner import *


# %% auto 0
__all__ = ['set_seed', 'Hook', 'Hooks', 'HooksCallback', 'append_stats', 'get_hist', 'get_min', 'ActivationStats']

# %% ../notebooks/10_activations.ipynb 3
def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# %% ../notebooks/10_activations.ipynb 20
class Hook():
    def __init__(self, module, func): self.hook = module.register_forward_hook(partial(func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

# %% ../notebooks/10_activations.ipynb 24
# Now I dont want to manage the individual hooks independently so time for a wrapper.
# This is a context manager that returns itself and registers our hooks. 
# The class will extend list to allow us to treat it as a list that stores the hooks
class Hooks(list):
    def __init__(self, modules, function): super().__init__([Hook(m, function) for m in modules])
    def __enter__(self, *args): return self # Returns self here so we can use `with as` to get our list of hooks
    def __exit__(self, *args): self.remove()
    def __del__(self): self.remove()
    def __defitem__(self, i): 
        self.remove()
        super().__delitem__(i)
    def remove(self): 
        for h in self: h.remove()

# %% ../notebooks/10_activations.ipynb 27
class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        fc.store_attr()
        super().__init__()
    
    def before_fit(self, learn):
        if self.mods: mods=self.mods
        else: mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training): self.hookfunc(*args, **kwargs)

    def after_fit(self, learn): self.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)


# %% ../notebooks/10_activations.ipynb 33
def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10)) 


# %% ../notebooks/10_activations.ipynb 35
def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p() # log here helps distribute the data in a useful visiual manner


# %% ../notebooks/10_activations.ipynb 38
def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)


# %% ../notebooks/10_activations.ipynb 41
class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): super().__init__(append_stats, mod_filter)
    def color_dim(self, figsize=(11,5)):
        fig,axs = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axs.flat, self):
            show_image(get_hist(h), ax, origin='lower')
    
    def dead_chart(self, figsize=(11,5)):
        fig,axs = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axs.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)
            
    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))
