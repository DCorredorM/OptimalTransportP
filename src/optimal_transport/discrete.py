from optimal_transport._base import *
import gurobipy as gp
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import pandas as pd
import numpy as np


class DiscreteOT:
	def __init__(self, alpha: DiscreteMeasure, beta: DiscreteMeasure, cost_function: Callable):
		self.cost_function = cost_function
		self.cost_matrix = {(i, j): cost_function(np.array([x, y]))
		                    for i, x in enumerate(alpha.support)
		                    for j, y in enumerate(beta.support)}
		self.alpha = alpha
		self.n = len(alpha.support)
		self.beta = beta
		self.m = len(beta.support)

		self.model, self.decision_variables = self._build_model()
		self.coupling = None

	def create_nodes(self, X=None, Y=None):
		X = self.alpha.support if X is None else X
		Y = self.beta.support if Y is None else Y
		M = np.array([[[x, y] for y in Y] for x in X])

		return M

	def _to_single_index(self, i, j):
		return j * self.n + i

	def _to_double_index(self, k, plus_one=False):
		j = k // self.n
		i = k % self.n
		if plus_one:
			return i + 1, j + 1
		else:
			return i, j

	def _build_model(self):
		model = gp.Model()
		model.setParam('OutputFlag', 0)
		n = range(len(self.alpha.support))
		m = range(len(self.beta.support))
		x = {}
		for (i, j), c in self.cost_matrix.items():
			x[i, j] = model.addVar(lb=0, ub=self.alpha.mass[i], obj=c, name=f'x({i},{j})')

		for i in n:
			model.addConstr(gp.quicksum(x[i, j] for j in m) == self.alpha.mass[i], name=f'out{i}')

		for j in m:
			model.addConstr(gp.quicksum(x[i, j] for i in n) == self.beta.mass[j], name=f'in{j}')

		model.update()
		return model, x

	def solve(self):
		self.model.optimize()
		# coupling = {key: var.x for key, var in self.decision_variables.items()}
		n = range(len(self.alpha.support))
		m = range(len(self.beta.support))

		coupling = np.array([[self.decision_variables[i, j].x for j in m] for i in n])

		self.coupling = coupling

	def plot(self, grid=True, hide_ticks=False, kind='hist'):
		n = 1000

		x, y = self.alpha.support, self.beta.support
		bins = max(len(x), len(y))
		x_y = [[(x[i], y[j])] * ceil(n * self.coupling[i, j]) for i in range(self.n) for j in range(self.m) if self.coupling[i, j] > 0]

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
		ax.set_xlabel(None)
		ax.set_ylabel(None)
		if grid:
			_, x_l = np.histogram(x, bins=bins)
			_, y_l = np.histogram(y, bins=bins)
			ax.hlines(y_l, *ax.get_xlim(), color='black', linewidth=.5, alpha=0.6)
			ax.vlines(x_l, *ax.get_ylim(), color='black', linewidth=.5, alpha=0.6)

	def plot_cost_function(self, resolution=200):
		xlist = np.linspace(-max(self.alpha.support), max(self.alpha.support), resolution)
		ylist = np.linspace(-max(self.beta.support), max(self.beta.support), resolution)

		X, Y = np.meshgrid(xlist, ylist)
		M = self.create_nodes(xlist, ylist)
		Z = np.apply_along_axis(self.cost_function, 2, M)
		fig, ax = plt.subplots(figsize=(9, 9))
		ax.contourf(X, Y, Z)


if __name__ == '__main__':
	import scipy.stats as sts
	n = 1000
	bins = 30

	x = np.concatenate([
		sts.gamma(1).rvs(int(n / 2)),
		sts.gamma(4, loc=5).rvs(int(n / 2))])
	y = sts.gamma(10).rvs(n)

	mass_x, support_x = np.histogram(x, bins=bins)
	support_x = (support_x[:-1] + support_x[1:]) / 2

	alpha = DiscreteMeasure(support=support_x, mass=mass_x / n)

	mass_y, support_y = np.histogram(y, bins=bins)
	support_y = (support_y[:-1] + support_y[1:]) / 2
	beta = DiscreteMeasure(support=support_y, mass=mass_y / n)

	def build_f(M):
		def f(x_bar):
			if isinstance(x_bar, list):
				x_bar = np.array(x_bar)
			return np.dot(np.dot(M, x_bar), x_bar.T)
		return f

	M = np.array([[1, 0],
	              [0, 1]])

	f = build_f(M)

	ot = DiscreteOT(alpha, beta, cost_function=f)
	ot.solve()
	# ot.plot()
	ot.plot_cost_function()
	plt.show()

