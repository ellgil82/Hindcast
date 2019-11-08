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
import datetime
import metpy
import metpy.calc

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

## Load data
def load_vars(year):
    try:
        melt_flux = iris.load_cube(filepath + year + '_land_snow_melt_flux.nc', 'Snow melt heating flux')  # W m-2
        melt_amnt = iris.load_cube(filepath + year + '_land_snow_melt_amnt.nc', 'Snowmelt')  # kg m-2 TS-1 (TS = 100 s)
        melt_amnt = iris.analysis.maths.multiply(melt_amnt, 108.) # 10800 s in 3 hrs / 100 s in a model timestep = 108 ==> melt amount per output timestep
        melt_rate = iris.load_cube(filepath+year+'_land_snow_melt_rate.nc', 'Rate of snow melt on land')
        orog = iris.load_cube(filepath + 'orog.nc')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(filepath + 'new_mask.nc')
        lsm = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    var_list = [melt_rate, melt_amnt, melt_flux, lsm, orog]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, np.ndim(i)-2, np.ndim(i)-1)
    vars_yr = {'melt_flux': melt_flux[:-4,0,:,:], 'melt_rate': melt_rate[:-4,0,:,:], 'melt_amnt': melt_amnt[:-4,0,:,:],
               'orog': orog, 'lsm': lsm,'lon': real_lon, 'lat': real_lat, 'year': year}
    return vars_yr

full_srs = load_vars('1998-2017')
#full_srs = load_vars('2016')

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

year_list = ['1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

def composite_plot(year_list):
    fig, ax = plt.subplots(7, 3, figsize=(8, 18))
    CbAx = fig.add_axes([0.25, 0.1, 0.5, 0.015])
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    total_melt = np.zeros((220,220))
    plot = 0
    for year in year_list:
        vars_yr = load_vars(year)
        c = ax[plot].pcolormesh(np.ma.masked_where(vars_yr['orog'].data > 50, a = np.cumsum(vars_yr['melt_amnt'].data, axis = 0)[-1]), vmin = 0, vmax = 300)
        ax[plot].contour(vars_yr['lsm'].data, colors = '#222222', lw = 2)
        ax[plot].contour(vars_yr['orog'].data, colors = '#222222', levels = [50])
        total_melt = total_melt + (np.cumsum(vars_yr['melt_amnt'].data, axis = 0)[-1])
        ax[plot].text(0.4, 1.1, s = year_list[plot], fontsize = 24,  color='dimgrey', transform = ax[plot].transAxes)
        plot = plot+1
    mean_melt_composite = np.ma.masked_where(vars_yr['orog'].data > 50, a =total_melt/len(year_list))
    ax[-1].contour(vars_yr['lsm'].data, colors='#222222', lw=2)
    ax[-1].pcolormesh(mean_melt_composite)
    ax[-1].contour(vars_yr['orog'].data, colors = '#222222', levels=[50])
    ax[-1].text(0., 1.1, s = '20 year mean', fontsize = 24,  color='dimgrey', transform = ax[-1].transAxes)
    cb = plt.colorbar(c, orientation = 'horizontal', cax = CbAx, ticks = [0,150,300])
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    #cb.ax.set_xticks([0,4,8])
    cb.set_label('Annual snow melt amount (kg m-$^{2}$)', fontsize = 24,  color='dimgrey', labelpad = 30)
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.15, hspace = 0.3, wspace = 0.05)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.eps', transparent = True)
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_all_years.png', transparent = True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_all_years.eps', transparent = True)
    plt.show()

#composite_plot(year_list)


def total_melt(srs):
    total_melt = np.zeros((220, 220))
    total_melt = total_melt + (np.cumsum(srs['melt_amnt'].data, axis=0)[-1])
    total_melt_masked = np.ma.masked_where(srs['orog'].data > 50, total_melt, copy = True)
    totm = total_melt_masked.sum()
    return totm, total_melt_masked

totm, totm_masked = total_melt(full_srs)


def totm_map(vars_yr, mean):
    fig, ax = plt.subplots(figsize=(8, 8))
    CbAx = fig.add_axes([0.25, 0.2, 0.5, 0.025])
    ax.axis('off')
    if mean == 'yes':
        c = ax.pcolormesh(totm_masked/20., vmin = 0,vmax = 300)
        xticks = [0,150,300]
        cb_lab = 'Annual mean snow melt amount (kg m$^{-2}$)'
    elif mean == 'no':
        c = ax.pcolormesh(totm_masked, vmin=0,vmax=5000)
        xticks = [0, 2500,5000]
        cb_lab = 'Cumulative snow melt amount (kg m$^{-2}$)'
    ax.contour(vars_yr['lsm'].data, colors='#222222')
    ax.contour(vars_yr['orog'].data, colors='#222222', levels=[50])
    plt.colorbar(c, cax = CbAx, orientation = 'horizontal')
    cb = plt.colorbar(c, orientation='horizontal', cax=CbAx, ticks=xticks, extend = "max")
    #cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    # cb.ax.set_xticks([0,4,8])
    cb.set_label(cb_lab, fontsize=24, color='dimgrey', labelpad=30)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.3, hspace=0.3, wspace=0.05)
    if host == 'bsl':
        if mean == 'yes':
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_cumulative_spatial_annual_mean.png', transparent=True)
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_cumulative_spatial_annual_mean.eps', transparent=True)
        else:
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_cumulative_spatial.png', transparent=True)
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_cumulative_spatial.eps', transparent=True)
    elif host == 'jasmin':
        if mean == 'yes':
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_spatial_annual_mean.png', transparent=True)
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_spatial_annual_mean.eps', transparent=True)
        else:
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_spatial.png',transparent=True)
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_cumulative_spatial.eps',transparent=True)
    plt.show()


#totm_map(full_srs, mean = 'yes')
#totm_map(full_srs, mean = 'no')

def melt_srs():
    if host == 'jasmin':
        filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
        ancil_path = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
        lsm_name = 'land_binary_mask'
    elif host == 'bsl':
        filepath = '/data/mac/ellgil82/hindcast/output/'
        ancil_path = filepath
        lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'
    try:
        orog = iris.load_cube(ancil_path + 'orog.nc', 'surface_altitude')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(ancil_path + 'new_mask.nc', lsm_name)
        LSM = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    tot_melt = iris.load_cube('1998-2017_land_snow_melt_amnt.nc')
    tot_melt = tot_melt[0]
    real_lon, real_lat = rotate_data(tot_melt, 1, 2)
    srs_vars = {'real_lon': real_lon, 'real_lat': real_lat, 'tot_melt': tot_melt, 'orog': orog, 'lsm': LSM}
    return srs_vars

# Create geographic location dictionaries necessary for plotting etc.

lon_index14, lat_index14, = find_gridbox(-67.01, -61.03, srs_vars['real_lat'], srs_vars['real_lon'])
lon_index15, lat_index15, = find_gridbox(-67.34, -62.09, srs_vars['real_lat'], srs_vars['real_lon'])
lon_index17, lat_index17, = find_gridbox(-65.93, -61.85, srs_vars['real_lat'], srs_vars['real_lon'])
lon_index18, lat_index18, = find_gridbox(-66.48272, -63.37105, srs_vars['real_lat'], srs_vars['real_lon'])

lat_dict = {'AWS14': -67.01,
            'AWS15': -67.34,
            'AWS17': -65.93,
            'AWS18': -66.48272}

lon_dict = {'AWS14': -61.03,
            'AWS15': -62.09,
            'AWS17': -61.85,
            'AWS18': -63.37105}

station_dict = {'AWS14_SEB_2009-2017_norp.csv': 'AWS14',
             'AWS15_hourly_2009-2014.csv': 'AWS15',
              'AWS17_SEB_2011-2015_norp.csv': 'AWS17',
                'AWS18_SEB_2014-2017_norp.csv': 'AWS18'}

def calc_melt_days(melt_var):
    melt = np.copy(melt_var)
    melt[melt > 0.] = 1. # if melt is detected at a timestep, this is 1 (i.e. True)
    out = melt.reshape(-1, 8, melt.shape[1], melt.shape[2]).sum(1)
    out = melt.sum(0)
    melt_days = np.count_nonzero(out, axis = 0)
    melt_days = np.ma.masked_where((full_srs['lsm'].data == 0.), out)
    #melt_days = np.ma.masked_where((full_srs['orog'].data > 50.), melt_days)
    return melt_days

#melt_max = iris.load_cube('1998-2017_land_snow_melt_amnt_daymax.nc')
#melt_max = melt_max[:,0,:,:]
melt_days = calc_melt_days(full_srs['melt_flux'].data)


def calc_melt_duration(melt_var):
    melt = np.copy(melt_var)
    melt_periods = np.count_nonzero(melt, axis = 0)
    melt_periods = melt_periods*3. # multiply by 3 to get number of hours per year (3-hourly data)
    return melt_periods

def chop_melt_yrs(melt_var):
    second_half_melt = np.zeros((220,220))
    melt_dur_2 = np.zeros((220,220))
    melt_days_2 = np.zeros((220, 220))
    for yr in year_list[1:-1]:
        df = pd.DataFrame()
        df['Timesrs'] = pd.date_range(datetime.datetime(int(yr), 1, 1, 0, 0, 0), datetime.datetime(int(yr), 12, 31, 23, 59, 59), freq='3H')
        months = [g for n,g in df.groupby(pd.Grouper(key = 'Timesrs', freq = 'M'))]
        first_half = pd.concat(months[:7])
        up_to = len(first_half) # find index to cut off at
        then_from = up_to
        melt_yr1 = iris.load_cube(filepath + yr + '_' + melt_var + '.nc')
        melt_yr2 = iris.load_cube(filepath + str(int(yr)+1) + '_' + melt_var + '.nc')
        melt_yr1 = melt_yr1[:,0,:,:]
        melt_yr2 = melt_yr2[:,0,:,:]
        # find cumulative melt up to end of July
        if melt_var == 'land_snow_melt_amnt':
            first_half_melt = np.cumsum(melt_yr1[:up_to, :,:].data, axis = 0)[-1]
        elif melt_var == 'land_snow_melt_flux':
            first_half_melt = np.mean(melt_yr1[:up_to, :, :].data, axis=0)[-1]
        melt_dur_1 = (np.count_nonzero(melt_yr1[:up_to, :,:].data, axis = 0)*3.) # multiply by 3 to get hours per year
        melt_days_1 = calc_melt_days(melt_yr1[:up_to, :,:].data)
        melt_seas = second_half_melt + first_half_melt
        if melt_var == 'land_snow_melt_flux':
            melt_seas = melt_seas/2.
        melt_dur_seas = melt_dur_2 + melt_dur_1
        melt_days_seas = melt_days_2 + melt_days_1
        # Save as a file
        melt_seas = iris.cube.Cube(melt_seas)
        melt_dur_seas = iris.cube.Cube(melt_dur_seas)
        melt_days_seas = iris.cube.Cube(melt_days_seas)
        if melt_var == 'land_snow_melt_amnt':
            iris.save(melt_seas, filepath + 'total_melt_amnt_during_' + yr[-2:] + '-' + str(int(yr)+1)[-2:]+ '_melt_season.nc')
        elif melt_var == 'land_snow_melt_flux':
            iris.save(melt_seas, filepath + 'mean_melt_flux_during_' + yr[-2:] + '-' + str(int(yr) + 1)[-2:] + '_melt_season.nc')
        iris.save(melt_dur_seas, filepath + 'melt_duration_during_' + yr[-2:] + '-' + str(int(yr)+1)[-2:]+ '_melt_season.nc')
        iris.save(melt_days_seas, filepath + 'number_of_melt_days_during_' + yr[-2:] + '-' + str(int(yr) + 1)[-2:] + '_melt_season.nc')
        if melt_var == 'land_snow_melt_amnt':
            second_half_melt = np.cumsum(melt_yr2[then_from:, :,:].data, axis = 0)[-1]
        elif melt_var == 'land_snow_melt_flux':
            second_half_melt = np.mean(melt_yr2[then_from:, :, :].data, axis=0)[-1]
        melt_dur_2 = (np.count_nonzero(melt_yr2[then_from:, :,:].data, axis = 0)*3.0)
        try:
            melt_days_2 = calc_melt_days(melt_yr2[then_from:, :, :].data)
        except:
            try:
                melt_days_2 = calc_melt_days(melt_yr2[then_from+2:, :, :].data)
            except:
                melt_days_2 = calc_melt_days(melt_yr2[then_from+4:, :, :].data)

chop_melt_yrs('land_snow_melt_amnt')

# Find all days where melt occurs during at least one timestep

#melt_duration = calc_melt_duration(full_srs['melt_amnt'].data)

def composite_melt_seasons():
    fig, ax = plt.subplots(6, 3, figsize=(8, 18))
    CbAx = fig.add_axes([0.25, 0.1, 0.5, 0.015])
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    file_list = []
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, 'number_of_melt_days_during_??-??_melt_season.nc'):
            file_list.append(file)
    first_file = 'number_of_melt_days_during_99-00_melt_season.nc'
    file_list = file_list[:-2]
    file_list = [first_file] + file_list
    total_melt = np.ma.masked_where((full_srs['lsm']== 0.), a = np.zeros((220,220)))
    for i in range(len(file_list)):
        melt = iris.load_cube(filepath + file_list[i])
        melt_masked = np.ma.masked_where((full_srs['orog'].data > 50), a =melt.data)
        c = ax[i].pcolormesh(melt_masked, vmin = 0, vmax = 200) #[0,:,:]), vmin = 0, vmax = 300)#, cmap = 'RdYlGn_r')
        ax[i].contour(full_srs['lsm'].data, colors = '#222222', lw = 2)
        ax[i].contour(full_srs['orog'].data, colors = '#222222', levels = [50])
        ax[i].text(0.4, 1.1, s= file_list[i][27:29]+'/'+file_list[i][30:32], fontsize=24, color='dimgrey', transform=ax[i].transAxes)
        total_melt = total_melt + np.ma.masked_where(full_srs['lsm'] == 0., a= melt.data)
    c = ax[-1].pcolormesh(np.ma.masked_where(full_srs['orog'].data > 50, a = np.ma.masked_where(full_srs['lsm'] == 0.,a = total_melt/18.)), vmin = 0, vmax = 200)#, cmap = 'RdYlGn_r')
    ax[-1].contour(full_srs['lsm'].data, colors='#222222', lw=2)
    ax[-1].contour(full_srs['orog'].data, colors='#222222', levels=[50])
    ax[-1].text(0., 1.1, s='20-year mean', fontsize=24, color='dimgrey', transform=ax[-1].transAxes)
    cb = plt.colorbar(c, orientation = 'horizontal', cax = CbAx, ticks = [0,100, 200], extend = 'max')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label('Annual melt_duration (days per year)', fontsize = 24,  color='dimgrey', labelpad = 30)
    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.15, hspace = 0.3, wspace = 0.05)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/melt_all_years.eps', transparent = True)
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Total_number_of_melt_days_per_year_viridis.png', transparent = True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Total_number_of_melt_days_per_year_viridis.eps', transparent = True)
    plt.show()
    return total_melt

total_melt = composite_melt_seasons()

def mean_melt_dur(total_melt, vars_yr):
    fig, ax = plt.subplots(figsize=(8, 8))
    CbAx = fig.add_axes([0.25, 0.2, 0.5, 0.025])
    ax.axis('off')
    c = ax.pcolormesh(np.ma.masked_where(vars_yr['orog'].data > 50, a= total_melt/18.), vmin = 0,vmax = 250, cmap ='RdYlGn_r')
    xticks = [0, 250]
    cb_lab = 'Annual mean melt duration(days per year)'
    ax.contour(vars_yr['lsm'].data, colors='#222222')
    ax.contour(vars_yr['orog'].data, colors='#222222', levels=[50])
    plt.colorbar(c, cax = CbAx, orientation = 'horizontal')
    cb = plt.colorbar(c, orientation='horizontal', cax=CbAx, ticks=xticks, extend = "max")
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label(cb_lab, fontsize=24, color='dimgrey', labelpad=30)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.3, hspace=0.3, wspace=0.05)
    if host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_duration_cf_Luckman14_&_Bevan18.png',transparent=True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/melt_duration_cf_Luckman14_&_Bevan18.eps',transparent=True)
    plt.show()

#mean_melt_dur(total_melt, full_srs)




calc_ts = [all_vars['Time_srs'][0]]
for i in range(58440):
    calc_ts.append(calc_ts[-1]+0.125)

