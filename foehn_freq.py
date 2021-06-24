import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import scipy.stats
import os
import iris
from datetime import time, datetime

filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'

# Load foehn frequency file
def foehn_freq(station, which):
	df = pd.read_csv(filepath + which + "_seasonal_foehn_frequency_" + station + "_1998_to_2017.csv", usecols = range(1,21), dtype = np.float64)
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
	SAM = pd.read_csv(filepath + '1998-2017_SAM_idx.csv', usecols = range(1,6), header = 2, skipfooter=3)
	SAM.index = range(1998,2018)
	stats_df = pd.DataFrame()
	inlet_p = []
	IS_p = []
	inlet_t = []
	IS_t = []
	t_18 = []
	p_18 = []
	t_17 = []
	p_17 = []
	t_15 = []
	p_15 = []
	t_14 = []
	p_14 = []
	for i in ["DJF", "MAM", "JJA", "SON", "ANN"]:
		t17, p17 = scipy.stats.pearsonr(mod17[i][:-2], SAM[i])
		t18, p18 = scipy.stats.pearsonr(mod18[i][:-2], SAM[i])
		t_18.append(t18)
		p_18.append(p18)
		t_17.append(t17)
		p_17.append(p17)
		inlet_p.append((p17 + p18)/2.)
		inlet_t.append((t17 + t18)/2.)
		t14, p14 = scipy.stats.pearsonr(mod14[i][:-2], SAM[i])
		t15, p15 = scipy.stats.pearsonr(mod15[i][:-2], SAM[i])
		IS_p.append((p14 + p15)/2.)
		IS_t.append((t14 + t15)/2.)
		t_15.append(t15)
		p_15.append(p15)
		t_14.append(t14)
		p_14.append(p14)
	stats_df['inlet r'] = pd.Series(inlet_t, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['inlet p'] = pd.Series(inlet_p, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['ice shelf r'] = pd.Series(IS_t, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['ice shelf p'] = pd.Series(IS_p, index = ["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 18 p'] = pd.Series(p_18, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 18 r'] = pd.Series(t_18, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 17 p'] = pd.Series(p_17, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 17 r'] = pd.Series(t_17, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 15 p'] = pd.Series(p_15, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 15 r'] = pd.Series(t_15, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 14 p'] = pd.Series(p_14, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df['AWS 14 r'] = pd.Series(t_14, index=["DJF", "MAM", "JJA", "SON", "ANN"])
	stats_df = stats_df.swapaxes(0,1)
	return stats_df, SAM

stats_df, SAM = SAM_correl()


stats_df.to_csv('Modelled_seasonal_SAM_correlations_inlet_v_ice_shelf.csv')

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

def plot_correls():
	seas_lens = {'DJF': 720, 'MAM': 736, 'JJA': 736, 'SON': 728, 'ANN': 2920}
	fig, axs = plt.subplots(figsize = (10,8))
	axs.spines['top'].set_visible(False)
	axs.spines['right'].set_visible(False)
	plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
	#axs.set(adjustable='box-forced', aspect='equal')
	axs.tick_params(axis='both', which='both', labelsize=24, width=2, length=8, color='dimgrey', labelcolor='dimgrey',
					pad=10)
	axs.yaxis.set_label_coords(-0.3, 0.5)
	axs.set_xlabel('Modelled foehn frequency (%)' , size=24, color='dimgrey', rotation=0, labelpad=10)
	axs.set_ylabel('Observed \nSAM index' , size=24, color='dimgrey', rotation=0, labelpad=10)
	colours = ['blue', 'green', 'orange', 'magenta', 'k']
	labs =["DJF", "MAM", "JJA", "SON", "ANN"]
	for i,j in enumerate(labs):
		axs.scatter((inlet_mod[j][:-2]/seas_lens[j])*100, SAM[j], color = colours[i], label = j)
		slope, intercept, r2, p, sterr = scipy.stats.linregress((inlet_mod[j][:-2]/seas_lens[j])*100, SAM[j])
		y_fit = slope*((inlet_mod[j][:-2]/seas_lens[j])*100) + intercept
		if j == 'ANN':
			axs.plot((inlet_mod[j][:-2] / seas_lens[j]) * 100, y_fit, linewidth = 3, color=colours[i])
		else:
			axs.plot((inlet_mod[j][:-2]/seas_lens[j])*100, y_fit, color=colours[i])
	# Legend
	lns = [Line2D([0], [0], color='blue', linewidth=4),
		   Line2D([0], [0], color='green', linewidth=4),
		   Line2D([0], [0], color='orange', linewidth=4),
		   Line2D([0], [0], color='magenta', linewidth=4),
		   Line2D([0], [0], color='k', linewidth=4)]
	lgd = axs.legend(lns, labs, bbox_to_anchor=(0.65, 0.45), loc=2, fontsize=20)
	frame = lgd.get_frame()
	frame.set_facecolor('white')
	for ln in lgd.get_texts():
		plt.setp(ln, color='dimgrey')
	lgd.get_frame().set_linewidth(0.0)
	plt.subplots_adjust(left=0.3, bottom = 0.25)
	plt.savefig(filepath + 'Scatter_foehn_v_SAM_seasonal.png')
	plt.savefig(filepath +  'Scatter_foehn_v_SAM_seasonal.eps')
	plt.show()

plot_correls()

def plot_foehn():
	fig, ax = plt.subplots(1,1,figsize = (16,8))
	plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
	ax.plot(range(1998, 2018), inlet_mod.ANN.iloc[:-2], color='orange', linewidth=2.5, label='Inlet stations, modelled')
	ax.plot(range(1998, 2018), iceshelf_mod.ANN.iloc[:-2], color='blue', linewidth=2.5,
			label='Ice shelf stations, modelled')
	ax.set_ylabel('Annual\nfoehn frequency', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	ax2 = ax.twinx()
	ax2.plot(range(1998, 2018), SAM.ANN, color='dimgrey', linewidth=2.5, linestyle=':')
	ax2.set_ylabel('SAM \nindex', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	for axs in ax, ax2:
		axs.spines['top'].set_visible(False)
		axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey')
		axs.set_xlim(1998, 2017)
		axs.set_xticks([1998, 2003, 2008, 2012, 2017])
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
	plt.subplots_adjust(left = 0.24, right = 0.85)
	plt.savefig('Annual_mean_modelled_foehn_frequency_vs_SAM.png')
	plt.savefig('Annual_mean_modelled_foehn_frequency_vs_SAM.eps')
	plt.show()

plot_foehn()

def foehn_time_srs():
	time_srs_is = np.ravel(iceshelf_mod[:-2].values[:,:-2])
	time_srs_in = np.ravel(inlet_mod[:-2].values[:,:-2])
	time_srsSAM = np.ravel(SAM.values[:,:-1])
	fig, ax = plt.subplots(1, 1, figsize=(16, 8))
	plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
	ax.plot(range(80), (time_srs_in/730.)*100, color='orange', linewidth=2.5, label='AWS 17, modelled')
	ax.plot(range(80), (time_srs_is/730.)*100, color='blue', linewidth=2.5, label='AWS 15, modelled')
	ax.set_ylabel('Seasonal foehn\nfrequency (%)', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	ax.set_ylim(0,30)
	ax2 = ax.twinx()
	plt.setp(ax2.spines.values(), linewidth=2, color='dimgrey')
	ax2.plot(range(80), time_srsSAM, color='dimgrey', linewidth=2.5, linestyle='--')
	ax2.set_ylabel('SAM \nindex', rotation=0, fontsize=28, labelpad=20, color='dimgrey')
	for axs in ax, ax2:
		axs.spines['top'].set_visible(False)
		axs.tick_params(axis='both', which='both', labelsize=24, width = 2, length = 8, color = 'dimgrey', labelcolor='dimgrey')
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
		   Line2D([0], [0], color='dimgrey', linewidth=2.5, linestyle='--')]
	labs = ['Ice shelf stations, modelled', 'Inlet stations, modelled', 'SAM index']
	lgd = ax.legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
	frame = lgd.get_frame()
	frame.set_facecolor('white')
	for ln in lgd.get_texts():
		plt.setp(ln, color='dimgrey')
	lgd.get_frame().set_linewidth(0.0)
	plt.subplots_adjust(left=0.24, right=0.85)
	plt.savefig('Total_time_srs_modelled_foehn_frequency_vs_SAM.png')
	plt.savefig('Total_time_srs_modelled_foehn_frequency_vs_SAM.eps')
	plt.show()

foehn_time_srs()

DJF = plt.plot(iceshelf_mod.DJF.values[:-2]/7.3, label = 'DJF')
MAM = plt.plot(iceshelf_mod.MAM.values[:-2]/7.3, label = 'MAM')
JJA = plt.plot(iceshelf_mod.JJA.values[:-2]/7.3, label = 'JJA')
SON = plt.plot(iceshelf_mod.SON.values[:-2]/7.3, label = 'SON')


import pymannkendall as mk
result = mk.seasonal_test(inlet_mod.SON)
plt.plot(df.index, df.IWP, label= 'IWP')
yr_mn = df.IWP.rolling(window = 7305, center = True).mean()
plt.plot(yr_mn, label = 'yearly rolling mean')
plt.show()

def calc_seas_SAM(seas):
	seas_lens = {'DJF': 1805,
				 'MAM': 1840,
				 'JJA': 1840,
				 'SON': 1820}
	SAM_full = pd.read_csv(filepath + 'Daily_mean_SAM_index_1998-2017.csv', usecols = ['SAM'], dtype = np.float64, header = 0, na_values = '*******')
	SAM_full.index = pd.date_range('1998-01-01', '2017-12-31', freq = 'D')
	months = [g for n, g in SAM_full.groupby(pd.TimeGrouper('M'))]
	SAM_seas = pd.Series()
	jan = np.arange(0,240,12)
	feb = np.arange(1,240,12)
	mar = np.arange(2,240,12)
	apr = np.arange(3,240,12)
	may = np.arange(4,240,12)
	jun = np.arange(5,240,12)
	jul = np.arange(6,240,12)
	aug = np.arange(7,240,12)
	sep = np.arange(8,240,12)
	oct = np.arange(9,240,12)
	nov = np.arange(10,240,12)
	dec = np.arange(11, 240, 12)
	for yr in range(20):
		if seas == 'DJF':
			SAM_seas = pd.concat((SAM_seas, months[dec[yr]], months[jan[yr]],months[feb[yr]] ))
		elif seas == 'MAM':
			SAM_seas = pd.concat((SAM_seas,months[mar[yr]], months[apr[yr]], months[may[yr]]))
		elif seas == 'JJA':
			SAM_seas = pd.concat((SAM_seas,months[jun[yr]], months[jul[yr]], months[aug[yr]]))
		elif seas == 'SON':
			SAM_seas = pd.concat((SAM_seas,months[sep[yr]], months[oct[yr]], months[nov[yr]]))
	SAM_seas = SAM_seas.values[:seas_lens[seas]-1, 1]
	return SAM_seas

#Calculate daily mean SAM index seasonally
MAM_SAM = calc_seas_SAM('MAM')
DJF_SAM = calc_seas_SAM('DJF')
SON_SAM = calc_seas_SAM('SON')
JJA_SAM = calc_seas_SAM('JJA')

LSM = iris.load_cube(filepath+'new_mask.nc')
lsm = LSM[0,0,:,:]

def calc_seas_FI(seas):
	#calc seasonal foehn index
	seas_FI = iris.load_cube(filepath + seas+ '_foehn_index.nc')
	seas_FI = seas_FI[:, 40:135, 90:155]
	lsm_3d = np.broadcast_to(lsm[ 40:135, 90:155].data, seas_FI.shape)
	seas_FI.data[lsm_3d == 0.] = np.nan
	mn_seas_FI = np.nanmean(seas_FI.data, axis = (1,2))
	return mn_seas_FI

MAM_FI = calc_seas_FI('MAM')
DJF_FI = calc_seas_FI('DJF')
JJA_FI = calc_seas_FI('JJA')
SON_FI = calc_seas_FI('SON')

#upsample SAM to get 3-hourly frequency by repeating daily totals
SAM_3hr_MAM = np.repeat(MAM_SAM, 8)
SAM_3hr_DJF = np.repeat(DJF_SAM, 8)
SAM_3hr_SON = np.repeat(SON_SAM, 8)
SAM_3hr_JJA = np.repeat(JJA_SAM, 8)

plt.scatter(SAM_3hr_MAM[:MAM_FI.shape[0]], MAM_FI, color = 'green', label = 'MAM')
plt.scatter(SAM_3hr_SON[:SON_FI.shape[0]], SON_FI, color = 'orange', label = 'SON')
plt.scatter(SAM_3hr_JJA[:JJA_FI.shape[0]], JJA_FI, color = 'blue', label = 'JJA')
plt.scatter(SAM_3hr_DJF[:DJF_FI.shape[0]], DJF_FI, color = 'magenta', label = 'DJF')
plt.legend()
plt.show()


foehn_df = pd.DataFrame()
foehn_df['Time'] = pd.date_range(datetime(1998,1,1,6,0,0), datetime(2017,12,31,15,0,0), freq = '3H')
df = pd.read_csv(filepath + 'ceda_archive/Foehn_freq_AWS18.csv')
freq = df['Unnamed: 0']
ff = np.zeros((58436))
ff[freq] = 1.
foehn_df['AWS18_foehn_freq'] = ff
df = pd.read_csv(filepath + 'ceda_archive/Foehn_freq_AWS14.csv')
freq = df['Unnamed: 0']
ff = np.zeros((58436))
ff[freq] = 1.
foehn_df['AWS14_foehn_freq'] = ff
df = pd.read_csv(filepath + 'ceda_archive/Foehn_freq_AWS15.csv')
freq = df['Unnamed: 0']
ff = np.zeros((58436))
ff[freq] = 1.
foehn_df['AWS15_foehn_freq'] = ff
foehn_df['sum_foehn'] = foehn_df.sum(axis = 1)
foehn_df.index = foehn_df['Time']
daily_foehn = foehn_df.resample('D').sum()
daily_foehn.to_csv(filepath + 'daily_foehn_frequency.csv')