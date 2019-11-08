import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import scipy.stats
import os

# Load foehn frequency file
def foehn_freq(station, which):
	df = pd.read_csv(which + "_seasonal_foehn_frequency_" + station + "_1998_to_2017.csv", usecols = range(1,21), dtype = np.float64)
	df["mean"] = df.mean(axis = 1)
	df["sum"] = df.sum(axis = 1)
	df.index = ["DJF", "MAM", "JJA", "SON", "ANN"]
	df = df.swapaxes(0,1)
	df["foehn_frac"] = (df["ANN"]/2920.)*100.
	df.iloc[-1,-1] = 0. # reset sum row to zero
	#plt.plot(df.mean)
	return df

mod18 = foehn_freq(station = "AWS18", which = "Modelled")
mod17 = foehn_freq(station = "AWS17", which = "Modelled")
mod15 = foehn_freq(station = "AWS15", which = "Modelled")
mod14 = foehn_freq(station = "AWS14", which = "Modelled")
obs18 = foehn_freq(station = "AWS18", which = "Observed")
obs17 = foehn_freq(station = "AWS17", which = "Observed")
obs15 = foehn_freq(station = "AWS15", which = "Observed")
obs14 = foehn_freq(station = "AWS14", which = "Observed")

# load foehn diagnosis method data
def sens_meth(station):
	df = pd.read_csv("Annual_foehn_frequency_modelled_" + station + ".csv", usecols = range(1,21), dtype = np.float64)
	df.index = ['Observed', 'Surface', 'Froude', 'Isentrope', 'Combo']
	return df

def sort_df():
	df14 = sens_meth('AWS14_SEB_2009-2017_norp.csv')
	df15 = sens_meth('AWS15_hourly_2009-2014.csv')
	df17 = sens_meth('AWS17_SEB_2011-2015_norp.csv')
	df18 = sens_meth('AWS18_SEB_2014-2017_norp.csv')
	obs = pd.DataFrame()
	surf = pd.DataFrame()
	froude = pd.DataFrame()
	isen = pd.DataFrame()
	combo = pd.DataFrame()
	for i, j, in zip(['AWS14', 'AWS15', 'AWS17', 'AWS18'], [df14, df15, df17, df18]):
		j = j.transpose()
		obs[i] = j['Observed']
		surf[i] = j['Surface']
		froude[i] = j['Froude']
		isen[i] = j['Isentrope']
		combo[i] = j['Combo']
	for s in [df14, df15, df17, df18]:
		s.fillna(0., inplace = True)
	ice_shelf = (df14 + df15)/2.
	inlet = (df17+df18)/2.
	return obs, surf, froude, isen, combo, ice_shelf, inlet

os.chdir('hindcast/output/alloutput')
obs, surf, froude, isen, combo, ice_shelf, inlet = sort_df()
t, p = scipy.stats.ttest_rel(inlet, ice_shelf, axis = 1)


inlet_mod = (mod18 + mod17)/2.
iceshelf_mod = (mod14 + mod15)/2.

# Run dependent samples t-test
# Are AWS 17 and AWS 15 statistically different?
t, p = scipy.stats.ttest_rel(mod17, mod15, axis = 1)
t, p = scipy.stats.ttest_rel(obs17, obs15, axis = 1)
t, p = scipy.stats.ttest_rel(inlet_mod, iceshelf_mod, axis = 1)

# Correlate with observed values?

# Correlate with SAM index
def SAM_correl():
	SAM = pd.read_csv('1998-2017_SAM_idx.csv', usecols = range(1,6), header = 2, skipfooter=3)
	SAM.index = range(1998,2018)
	stats_df = pd.DataFrame()
	inlet_p = []
	IS_p = []
	inlet_t = []
	IS_t = []
	for i in ["DJF", "MAM", "JJA", "SON", "ANN"]:
		#t, p = scipy.stats.ttest_rel(obs17[i][:-2], SAM[i])
		#stats_df['obs17'] = pd.Series[p]
		t, p = scipy.stats.pearsonr(mod17[i][:-2], SAM[i])
		inlet_p.append(p)
		inlet_t.append(t)
		#t, p = scipy.stats.ttest_rel(obs15[i][:-2], SAM[i])
		#stats_df['obs15'] = pd.Series[p]
		t, p = scipy.stats.pearsonr(mod15[i][:-2], SAM[i])
		IS_p.append(p)
		IS_t.append(t)
	stats_df['inlet r'] = pd.Series(inlet_t, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['inlet p'] = pd.Series(inlet_p, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['ice shelf r'] = pd.Series(IS_t, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['ice shelf p'] = pd.Series(IS_p, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df = stats_df.swapaxes(0,1)
	return stats_df, SAM

stats_df, SAM = SAM_correl()

stats_df.to_csv('Modelled_seasonal_SAM_correlations_inlet_v_ice_shelf.csv')

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

def plot_foehn():
	fig, ax = plt.subplots(1,1,figsize = (16,8))
	plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
	ax.plot(range(1998,2018), inlet_mod.ANN.iloc[:-2], color = 'orange', linewidth = 2.5, label = 'Inlet stations, modelled')
	ax.plot(range(1998,2018), iceshelf_mod.ANN.iloc[:-2], color = 'blue', linewidth = 2.5, label = 'Ice shelf stations, modelled')
	ax.set_ylabel('Annual\nfoehn frequency', rotation = 0,fontsize=28, labelpad=20, color='dimgrey')
	ax2  = ax.twinx()
	ax2.plot(range(1998,2018), SAM.ANN, color = 'dimgrey', linewidth = 2.5, linestyle = ':')
	ax2.set_ylabel('SAM \nindex', rotation = 0, fontsize=28, labelpad=20, color='dimgrey')
	for axs in ax, ax2:
		axs.spines['top'].set_visible(False)
		axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey')
		axs.set_xlim(1998,2017)
		axs.set_xticks([1998, 2003, 2008, 2012, 2017])
	[l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
	ax.yaxis.set_label_coords(-0.22, 0.5)
	ax2.yaxis.set_label_coords(1.15, 0.62)
	lns = [Line2D([0], [0], color='blue', linewidth=2.5),
		   Line2D([0], [0], color='orange', linewidth=2.5),
		   Line2D([0], [0], color='dimgrey', linewidth=2.5, linestyle = ':')]
	labs = ['Ice shelf stations, modelled', 'Inlet stations, modelled', 'SAM index']
	lgd = ax.legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
	frame = lgd.get_frame()
	frame.set_facecolor('white')
	for ln in lgd.get_texts():
		plt.setp(ln, color='dimgrey')
	lgd.get_frame().set_linewidth(0.0)
	plt.subplots_adjust(left = 0.24, right = 0.85)
	plt.savefig('Annual_mean_modelled_foehn_frequency_vs_SAM.png', transparent = True)
	plt.savefig('Annual_mean_modelled_foehn_frequency_vs_SAM.eps', transparent = True)
	plt.show()

plot_foehn()

def foehn_time_srs():
	time_srs_is = np.ravel(iceshelf_mod[:-2].values[:,:-2])
	time_srs_in = np.ravel(inlet_mod[:-2].values[:,:-2])
	time_srsSAM = np.ravel(SAM.values[:,:-1])
	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
	ax.plot(range(80),time_srs_in, color='orange', linewidth=2.5, label='AWS 17, modelled')
	ax.plot(range(80), time_srs_is, color='blue', linewidth=2.5, label='AWS 15, modelled')
	ax.set_ylabel('Seasonal \nfoehn \nfrequency', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	ax.set_ylim(0,180)
	ax2 = ax.twinx()
	ax2.plot(range(80), time_srsSAM, color='dimgrey', linewidth=2.5, linestyle=':')
	ax2.set_ylabel('SAM \nindex', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	for axs in ax, ax2:
		axs.spines['top'].set_visible(False)
		axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey')
		axs.set_xlim(0,79)
		labels = [item.get_text() for item in axs.get_xticklabels()]
		labels[1] = '2000'
		labels[3] = '2005'
		labels[5] = '2010'
		labels[7] = '2015'
		axs.set_xticklabels(labels)
	[l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
	ax.yaxis.set_label_coords(-0.22, 0.5)
	ax2.yaxis.set_label_coords(1.15, 0.62)
	lns = [Line2D([0], [0], color='blue', linewidth=2.5),
		   Line2D([0], [0], color='orange', linewidth=2.5),
		   Line2D([0], [0], color='dimgrey', linewidth=2.5, linestyle=':')]
	labs = ['Ice shelf stations, modelled', 'Inlet stations, modelled', 'SAM index']
	lgd = ax.legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
	frame = lgd.get_frame()
	frame.set_facecolor('white')
	for ln in lgd.get_texts():
		plt.setp(ln, color='dimgrey')
	lgd.get_frame().set_linewidth(0.0)
	plt.subplots_adjust(left=0.24, right=0.85)
	plt.savefig('Total_time_srs_modelled_foehn_frequency_vs_SAM.png', transparent=True)
	plt.savefig('Total_time_srs_modelled_foehn_frequency_vs_SAM.eps', transparent=True)
	plt.show()

foehn_time_srs()
