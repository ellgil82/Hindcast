# Define where the script is running
host = 'jasmin'

# Import packages
import iris
import iris.plot as iplt
import sys
import numpy as np
import pandas as pd
import os
import fnmatch
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.lines import Line2D
if host == 'jasmin':
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    sys.path.append('/users/ellgil82/scripts/Tools/')

from tools import compose_date, compose_time, find_gridbox
from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import time
from sklearn.metrics import mean_squared_error
from datetime import datetime
import metpy
import metpy.calc

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

def load_time_srs():
    cloud_frac = iris.load_cube(filepath + '1998-2017_cl_frac_daymn.nc')
    IWP = iris.load_cube(filepath + '1998-2017_total_column_ice_daymn.nc')
    LWP = iris.load_cube(filepath + '1998-2017_total_column_liquid_daymn.nc')
    WVP = iris.load_cube(filepath + '1998-2017_total_column_vapour_daymn.nc')
    lsm = iris.load_cube(filepath + 'new_mask.nc')
    orog = iris.load_cube(filepath + 'orog.nc')
    orog = orog[0, 0, :, :]
    lsm = lsm[0, 0, :, :]
    # Rotate data onto standard lat/lon grid
    for i in [ cloud_frac, IWP, LWP, WVP, orog, lsm]:
        real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
    var_dict = {'IWP': IWP[:,0,:,:], 'LWP': LWP[:,0,:,:], 'WVP': WVP[:,0,:,:],  'cl_frac': cloud_frac[:,0,:,:],
                'Time_srs': cloud_frac.coord('time').points,  'lat': real_lat, 'lon': real_lon, 'orog': orog, 'lsm': lsm}
    return var_dict

all_vars = load_time_srs()

lon_index14, lat_index14 = find_gridbox(-67.01, -61.03, all_vars['lat'], all_vars['lon'])
lon_index15, lat_index15 = find_gridbox(-67.34, -62.09, all_vars['lat'], all_vars['lon'])
lon_index17, lat_index17 = find_gridbox(-65.93, -61.85, all_vars['lat'], all_vars['lon'])
lon_index18, lat_index18 = find_gridbox(-66.48272, -63.37105, all_vars['lat'], all_vars['lon'])

lat_dict = {'AWS14': lat_index14,
            'AWS15': lat_index15,
            'AWS17': lat_index17,
            'AWS18': lat_index18}

lon_dict = {'AWS14': lon_index14,
            'AWS15': lon_index15,
            'AWS17': lon_index17,
            'AWS18': lon_index18}

station_dict = {'AWS14_SEB_2009-2017_norp.csv': 'AWS14',
             'AWS15_hourly_2009-2014.csv': 'AWS15',
              'AWS17_SEB_2011-2015_norp.csv': 'AWS17',
                'AWS18_SEB_2014-2017_norp.csv': 'AWS18'}

def apply_Larsen_mask(var):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = all_vars['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[all_vars['orog'].data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    # Apply mask to variable requested
    var_masked = np.ma.masked_array(var, mask=np.broadcast_to(Larsen_mask, var.shape)).mean(axis=(1, 2))
    return var_masked

cl_masked = np.mean(all_vars['cl_frac'].data[:, 40:140, 85:155], axis = (1,2))
IWP_masked = np.mean(all_vars['IWP'].data[:, 40:140, 85:155], axis = (1,2))
LWP_masked = np.mean(all_vars['LWP'].data[:, 40:140, 85:155], axis = (1,2))
WVP_masked = np.mean(all_vars['WVP'].data[:, 40:140, 85:155], axis = (1,2))
df = pd.DataFrame([pd.Series(cl_masked[:7305]), pd.Series(WVP_masked[:7305]), pd.Series(LWP_masked[:7305]), pd.Series(IWP_masked[:7305])], index = ['cl', 'WVP', 'LWP', 'IWP'])
df = df.transpose()
df['datetime'] = pd.date_range(datetime(1998,1,1,0,0,0),datetime(2017,12,31,23,59,59), freq = 'D')
df = df.set_index('datetime')

import pymannkendall as mk
result = mk.seasonal_test(df.cl)
plt.plot(df.index, df.IWP, label= 'IWP')
yr_mn = df.IWP.rolling(window = 7305, center = True).mean()
plt.plot(yr_mn, label = 'yearly rolling mean')
plt.show()


def calc_percent_cloudy(cl_masked):
    ''' Calculate percentage of time where ice-shelf integrated cloud fraction falls into one of the following
    categories, as defined in Kay et al. (2008) doi: 10.1029/2011RG000363:

        1. 'Clear': cloud fraction < 0.31
        2. 'Scattered cloud': 0.31 < cloud fraction > 0.75
        3. ' Broken cloud': 0.75 < cloud fraction > 1.0
        4. 'Overcast': cloud fraction == 1.0

    '''
    clear = np.copy(cl_masked) # create ice-shelf averaged time series of cloud cover
    clear[clear < 0.31] = 0. # anything below threshold is "clear"
    clear = np.logical_not(clear) # invert 0s to calculate number of non-zero timesteps
    clear_pct = (np.float(np.count_nonzero(clear))/clear.shape[0])*100
    scatt_cl = np.copy(cl_masked)
    scatt_cl[scatt_cl <= 0.31] = 0.
    scatt_cl[scatt_cl > 0.75] = 0.
    scatt_pct = (np.float(np.count_nonzero(scatt_cl))/scatt_cl.shape[0])*100
    broken_cl = np.copy(cl_masked)
    broken_cl[broken_cl >= 1.] = 0.
    broken_cl[broken_cl <= 0.75] = 0.
    broken_pct = (np.float(np.count_nonzero(broken_cl))/broken_cl.shape[0])*100
    overcast = np.copy(cl_masked)
    overcast[overcast < 1.] = 0.
    overcast_pct = (np.float(np.count_nonzero(overcast))/overcast.shape[0])*100
    cloudy_pct = scatt_pct + broken_pct + overcast_pct
    return overcast_pct, broken_pct, scatt_pct, cloudy_pct, clear_pct

def seas_mean_cloud():
    timesrs = pd.date_range(datetime(1997, 12, 31, 0, 0, 0), datetime(2017, 12, 31, 23, 59, 59), freq='D')
    # load time series into dateframe
    cl_df = pd.DataFrame()
    cl_df['cloud fraction'] = cl_masked
    cl_df['IWP'] = IWP_masked
    cl_df['LWP'] = LWP_masked
    cl_df['WVP'] = WVP_masked
    cl_df.index = timesrs
    # group into seasons
    months = [g for n, g in cl_df.groupby(pd.TimeGrouper('M'))]
    cl_DJF = pd.concat((months[11], months[0], months[1]))
    cl_MAM = pd.concat((months[2], months[3], months[4]))
    cl_JJA = pd.concat((months[5], months[6], months[7]))
    cl_SON = pd.concat((months[8], months[9], months[10]))
    return cl_df, cl_DJF, cl_MAM, cl_JJA, cl_SON

def make_table():
    cl_df, cl_DJF, cl_MAM, cl_JJA, cl_SON = seas_mean_cloud()
    seas_cloudiness = pd.DataFrame()
    for i, j in zip([cl_df, cl_DJF, cl_MAM, cl_JJA, cl_SON], ['ANN', 'DJF', 'MAM', 'JJA', 'SON']):
        overcast_pct, broken_pct, scatt_pct, cloudy_pct, clear_pct = calc_percent_cloudy(i['cloud fraction'])
        seas_cloudiness[j] = pd.Series([overcast_pct, broken_pct, scatt_pct, cloudy_pct, clear_pct])
    seas_cloudiness.index = ['"Overcast" = 1.', '1. < "Broken" > 0.75', '0.75 < "Scattered" > 0.31', '"Cloudy" > 0.31', '"Clear" < 0.31']
    seas_cloudiness.to_csv(filepath + 'Seasonal_cloudiness_stats_Kay_definitions_ice_shelf_integrated.csv')
    return seas_cloudiness

seas_cloudiness = make_table()

def find_seasmean_values(file):
    seas_mean = iris.load_cube(filepath + file)
    seas_mean = seas_mean[:,0]
    if file == '1998-2017_seasmean_LWP.nc' or file == '1998-2017_seasmean_IWP.nc':
        seas_mean.convert_units('g kg-1')
    DJF = seas_mean[0::4]
    MAM = seas_mean[1::4]
    JJA = seas_mean[2::4]
    SON = seas_mean[3::4]
    return DJF, MAM, JJA, SON

def find_seas_all_values(var):
    DJF = iris.load_cube(filepath + '1998-2017_DJF_' + var + '.nc')
    MAM = iris.load_cube(filepath + '1998-2017_MAM_' + var + '.nc')
    JJA = iris.load_cube(filepath + '1998-2017_JJA_' + var + '.nc')
    SON = iris.load_cube(filepath + '1998-2017_SON_' + var + '.nc')
    if var == 'total_column_vapour' or var == 'total_column_ice' or var == 'total_column_liquid':
        for seas in [DJF, MAM, JJA, SON]:
            seas.convert_units('g kg-1')
    return DJF[:,0], MAM[:,0], JJA[:,0], SON[:,0]


DJF, MAM, JJA, SON = find_seasmean_values('1998-2017_seasmean_cl_frac.nc')

lims = {'cloud_fraction': (0.6, 0.95, 'Seasonal mean cloud fraction'),
        'IWP': (0.05, 0.25, 'Ice water path (g kg$^{-1}$)'),
        'LWP': (0.01, 0.15, 'Liquid water path (g kg$^{-1}$)'),
        'WVP': (2.5, 10.0, 'Water vapour path (g kg$^{-1}$)')}

lsm = iris.load_cube(filepath + 'new_mask.nc')
orog = iris.load_cube(filepath + 'orog.nc')
orog = orog[0, 0, :, :]
lsm = lsm[0, 0, :, :]

for i in [lsm, orog]:
    real_lon, real_lat = rotate_data(i, 0,1)

all_vars = {'lsm': lsm, 'orog': orog, 'lat': real_lat, 'lon': real_lon}

def plot_seas_cl_maps(var_name):
    fig, ax = plt.subplots(2,2, figsize = (8,10))
    CbAx = fig.add_axes([0.25, 0.15, 0.5, 0.02])
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    plot = 0
    for i, j in zip([DJF, MAM, JJA, SON], ['DJF', 'MAM', 'JJA', 'SON']):
        c = ax[plot].pcolormesh(np.mean(i.data, axis = 0), vmin = lims[var_name][0], vmax = lims[var_name][1])
        ax[plot].contour(all_vars['lsm'].data, colors = '#222222', lw = 2)
        ax[plot].contour(all_vars['orog'].data, colors = '#222222', levels = [50])
        ax[plot].text(0.4, 1.1, s= j, fontsize=24, color='dimgrey', transform=ax[plot].transAxes)
        plot = plot + 1
    cb = plt.colorbar(c, orientation = 'horizontal', cax = CbAx, ticks = [lims[var_name][0], lims[var_name][1]], extend = 'both')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label(lims[var_name][2], fontsize = 24,  color='dimgrey', labelpad = 20)
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.92, bottom = 0.2, hspace = 0.2, wspace = 0.05)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/seasonal_cloud_maps_' + var_name + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/seasonal_cloud_maps_' + var_name + '.eps')
    plt.show()

DJF_frac, MAM_frac, JJA_frac, SON_frac = find_seasmean_values('1998-2017_seasmean_cl_frac.nc')

plot_seas_cl_maps('cloud_fraction')

DJF_IWP, MAM_IWP, JJA_IWP, SON_IWP = find_seasmean_values('1998-2017_seasmean_IWP.nc')

plot_seas_cl_maps('IWP')

DJF_LWP, MAM_LWP, JJA_LWP, SON_LWP = find_seasmean_values('1998-2017_seasmean_LWP.nc')

plot_seas_cl_maps('LWP')

DJF_WVP, MAM_WVP, JJA_WVP, SON_WVP = find_seasmean_values('1998-2017_seasmean_WVP.nc')

plot_seas_cl_maps('WVP')

# Find differences between 90th and 10th percentiles

IWP_difs = np.percentile(a = surf['IWP'].data, q = 90, axis = 0) - np.percentile(a = surf['IWP'].data, q = 10, axis = 0)
LWP_difs = np.percentile(a = surf['LWP'].data, q = 90, axis = 0) - np.percentile(a = surf['LWP'].data, q = 10, axis = 0)
WVP_difs = np.percentile(a = surf['WVP'].data, q = 90, axis = 0) - np.percentile(a = surf['WVP'].data, q = 10, axis = 0)
cl_difs = np.percentile(a = surf['cl_cover'].data, q = 90, axis = 0) - np.percentile(a = surf['cl_cover'].data, q = 10, axis = 0)

title_dict = {'SW_net': 'SW$_{net}$', 'LW_net': 'LW$_{net}$', 'HL': 'H$_{L}$', 'HS':'H$_{S}$', 'E_tot': 'E$_{tot}$',
              'melt_flux': 'E$_{melt}$', 'IWP': 'IWP', 'LWP': 'LWP', 'cl_cover': 'cloud cover', 'WVP': 'WVP', IWP_difs: 'IWP',
              LWP_difs: 'LWP', WVP_difs: 'WVP', cl_difs: 'cloud cover'}


### Caption: difference between the 90th and 10th percentiles of water vapour path, ice water path and liquid water path
## during the entire hindcast period.

def plot_cloud_difs(var1, var2, var3):
    fig, axs = plt.subplots(1,3, frameon=False, figsize=( 20, 6))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var1, var2, var3]
    lims = {'WVP': 8, 'IWP': 1., 'LWP': 0.5}
    for i, j in zip(axs, ['WVP', 'IWP', 'LWP']):
        i.set_title(j, color = 'dimgrey', fontsize = 34)
    plt.axis = 'off'
    #for ax in [axs[1], axs[2]]:
    #    ax.yaxis.tick_right()
    for ax, var, name in zip(axs, var_list, ['WVP', 'IWP', 'LWP']):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
        PlotLonMin = np.min(all_vars['lon'])
        PlotLonMax = np.max(all_vars['lon'])
        PlotLatMin = np.min(all_vars['lat'])
        PlotLatMax = np.max(all_vars['lat'])
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
        plt.sca(ax)
        plt.xticks(XTicks, XTickLabels)
        ax.set_xlim(PlotLonMin, PlotLonMax)
        ax.tick_params(which='both', pad=10, labelsize = 24, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.yticks(YTicks, YTickLabels)
        ax.set_ylim(PlotLatMin, PlotLatMax)
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
        xlon, ylat = np.meshgrid(all_vars['lon'], all_varrs['lat'])
        cf_var = var
        c = ax.pcolormesh(xlon, ylat, cf_var, vmin = 0, vmax = lims[name], cmap='Blues')#,  vmax = 1, vmin=-6., vmax=3., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        coast = ax.contour(xlon, ylat, all_vars['lsm'].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, all_vars['orog'].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
        CBarXTicks = [0, lims[name]]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    #CBAxes = fig.add_axes([0.2, 0.15, 0.6, 0.015])
        CBar = plt.colorbar(c,  extend='max',ticks=CBarXTicks)# cax=CBAxes, )orientation='horizontal',
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
        CBar.outline.set_linewidth(2)
        CBar.set_label('Difference \n(g kg$^{-1}$)', rotation = 0, y=1.25, fontsize=24, labelpad=-40, color='dimgrey')
    plt.subplots_adjust(bottom = 0.2, top = 0.8, hspace = 0.1, left = 0.1, wspace = 0.35, right = 0.9)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Cloud_percentile_difs_subplot.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Cloud_percentile_difs_subplot.eps')
    plt.show()

plot_cloud_difs(WVP_difs, IWP_difs, LWP_difs)


# Difference between the 90th and 10th percentiles of seasonal mean water vapour path, ice water path and liquid water path
## during the entire hindcast period for < season >.
def plot_seas_cl_difs(var1, var2, var3, season):
    fig, axs = plt.subplots(1,3, frameon=False, figsize=( 20, 6))
    axs = axs.flatten()
    fig.patch.set_visible(False)
    var_list = [var1, var2, var3]
    lims = {'WVP': 8, 'IWP': 1, 'LWP': 0.5}
    for i, j in zip(axs, ['WVP', 'IWP', 'LWP']):
        i.set_title(j, color = 'dimgrey', fontsize = 34)
    plt.axis = 'off'
    #for ax in [axs[1], axs[2]]:
    #    ax.yaxis.tick_right()
    for ax, var, name in zip(axs, var_list, ['WVP', 'IWP', 'LWP']):
        ax.axis = 'off'
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
        PlotLonMin = np.min(all_vars['lon'])
        PlotLonMax = np.max(all_vars['lon'])
        PlotLatMin = np.min(all_vars['lat'])
        PlotLatMax = np.max(all_vars['lat'])
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
        plt.sca(ax)
        plt.xticks(XTicks, XTickLabels)
        ax.set_xlim(PlotLonMin, PlotLonMax)
        ax.tick_params(which='both', pad=10, labelsize = 24, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.yticks(YTicks, YTickLabels)
        ax.set_ylim(PlotLatMin, PlotLatMax)
        ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                       tick2On=False)
        xlon, ylat = np.meshgrid(all_vars['lon'], all_vars['lat'])
        cf_var = np.percentile(a = var, q = 90, axis = 0) - np.percentile(a = var, q = 10, axis = 0)
        c = ax.pcolormesh(xlon, ylat, cf_var, vmin = 0, vmax = lims[name], cmap='Blues')#,  vmax = 1, vmin=-6., vmax=3., zorder=1)  # latlon=True, transform=ccrs.PlateCarree(),
        coast = ax.contour(xlon, ylat, all_vars['lsm'].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, all_vars['orog'].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
        CBarXTicks = [0, lims[name]]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    #CBAxes = fig.add_axes([0.2, 0.15, 0.6, 0.015])
        CBar = plt.colorbar(c,  extend='max',ticks=CBarXTicks)# cax=CBAxes, )orientation='horizontal',
        CBar.solids.set_edgecolor("face")
        CBar.outline.set_edgecolor('dimgrey')
        CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
        CBar.outline.set_linewidth(2)
        CBar.set_label('Difference \n(g kg$^{-1}$)', rotation = 0, y=1.25, fontsize=24, labelpad=-40, color='dimgrey')
    plt.subplots_adjust(bottom = 0.2, top = 0.8, hspace = 0.1, left = 0.1, wspace = 0.35, right = 0.9)
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Cloud_percentile_difs_subplot_' + season + '.png')
    plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Cloud_percentile_difs_subplot_' + season + '.eps')
    plt.show()


DJF_WVP, MAM_WVP, JJA_WVP, SON_WVP = find_seas_all_values('total_column_vapour')
DJF_IWP, MAM_IWP, JJA_IWP, SON_IWP = find_seas_all_values('total_column_ice')
DJF_LWP, MAM_LWP, JJA_LWP, SON_LWP = find_seas_all_values('total_column_liquid')
DJF_cl_frac, MAM_cl_frac, JJA_cl_frac, SON_cl_frac = find_seas_all_values('cl_frac')

plot_seas_cl_difs(DJF_WVP.data, DJF_IWP.data, DJF_LWP.data, season = 'DJF')
plot_seas_cl_difs(MAM_WVP.data, MAM_IWP.data, MAM_LWP.data, season = 'MAM')
plot_seas_cl_difs(JJA_WVP.data, JJA_IWP.data, JJA_LWP.data, season = 'JJA')
plot_seas_cl_difs(SON_WVP.data, SON_IWP.data, SON_LWP.data, season = 'SON')
