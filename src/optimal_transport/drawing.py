import matplotlib.pyplot as plt
import scipy.stats as sts
import seaborn as sns
import numpy as np
import pandas as pd


def marginals(kind='kde', bins=20):
	n = 1000
	x = np.concatenate([
		sts.gamma(1).rvs(int(n / 2)),
		sts.gamma(4, loc=5).rvs(int(n / 2))])
	y = sts.gamma(10).rvs(n)
	if kind == 'hist':
		marginal_kws = dict(bins=bins)
		joint_kws = dict(bins=bins)
	else:
		marginal_kws = dict()
		joint_kws = dict()

	jp = sns.jointplot(data=pd.DataFrame({'x': x, 'y': y}), x='x', y='y', legend=False, kind=kind, fill=True, marginal_kws=marginal_kws, joint_kws=joint_kws)

	jp.ax_joint.tick_params(left=False, right=False, labelleft=False,
	                        labelbottom=False, bottom=False)
	jp.ax_marg_x.tick_params(left=False, right=False, labelleft=False,
	                         labelbottom=False, bottom=False)
	jp.ax_marg_y.tick_params(left=False, right=False, labelleft=False,
	                         labelbottom=False, bottom=False)
	# jp.ax_joint.axis("off")
	ax = jp.ax_joint
	ax.set_xlabel(None)
	ax.set_ylabel(None)
	if kind == 'hist':
		_, x_l = np.histogram(x, bins=bins)
		_, y_l = np.histogram(y, bins=bins)
		ax.hlines(y_l, *ax.get_xlim(), color='black', linewidth=.5, alpha=0.6)
		ax.vlines(x_l, *ax.get_ylim(), color='black', linewidth=.5, alpha=0.6)

	return jp


def marginals_sbs():
	n = 100
	x = np.concatenate([
		sts.gamma(1).rvs(int(n / 2)),
		sts.gamma(4, loc=5).rvs(int(n / 2))])

	y = sts.gamma(10).rvs(n) + 30

	df = pd.DataFrame({'mu1': x, 'mu2': y})
	ax = sns.kdeplot(data=df, fill=True, legend=False)
	plt.show()
	ax.axis("off")


if __name__ == '__main__':
	np.random.seed(754)
	marginals('kde', bins=30)
	plt.show()