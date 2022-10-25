from math import ceil
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns


class OptimalTransportSolution:
    def __init__(self, coupling: np.ndarray, instance: 'OptimalTransport'):
        self.instance: 'OptimalTransport' = instance
        self.coupling = coupling
        
    def draw_slice(self, t, **kwargs):
        idx = list(range(len(self.coupling.shape)))
        idx.remove(t)
        idx.remove(t + 1)
        
        mat = self.coupling.sum(axis=tuple(idx))
        
        self._mat_show(
            matrix=mat,
            x=self.instance.mu[t].support,
            y=self.instance.mu[t+1].support,
            time=t,
            **kwargs)
    
    def _mat_show(
            self,
            matrix: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            time: int,
            grid=True,
            hide_ticks=False,
            kind='hist'
    ):
        
        bins = max(len(x), len(y))
        
        # This sns function plots lists of pairs (x, y), here we try to create a sample that represents the actual
        # distribution. That is why we create enough samples and multiply them by the frequency given by the coupling.
        n = 1000
        x_y = [
            [(x[i], y[j])] * ceil(n * matrix[i, j])
            for i in range(len(x)) for j in range(len(y)) if matrix[i, j] > 0
        ]
        
        x_y = sum(x_y, [])
        x_d, y_d = tuple(zip(*x_y))
    
        marginal_kws = dict(bins=bins)
        joint_kws = dict(bins=bins)
    
        jp = sns.jointplot(data=pd.DataFrame({'x': x_d, 'y': y_d}),
                           x='x',
                           y='y',
                           legend=False,
                           kind=kind,
                           fill=True,
                           marginal_kws=marginal_kws,
                           joint_kws=joint_kws,
                           height=8)
    
        if hide_ticks:
            jp.ax_joint.tick_params(left=False, right=False, labelleft=False,
                                    labelbottom=False, bottom=False)
            jp.ax_marg_x.tick_params(left=False, right=False, labelleft=False,
                                     labelbottom=False, bottom=False)
            jp.ax_marg_y.tick_params(left=False, right=False, labelleft=False,
                                     labelbottom=False, bottom=False)
        ax = jp.ax_joint
        ax.set_xlabel(r'$S_'+f'{time+1}' + r'$')
        ax.set_ylabel(r'$S_'+f'{time}' + r'$')
        if grid:
            _, x_l = np.histogram(x, bins=bins)
            _, y_l = np.histogram(y, bins=bins)
            ax.hlines(y_l, *ax.get_xlim(), color='black', linewidth=.5, alpha=0.6)
            ax.vlines(x_l, *ax.get_ylim(), color='black', linewidth=.5, alpha=0.6)