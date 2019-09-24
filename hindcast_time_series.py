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
    ancil_path = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
    lsm_name = 'land_binary_mask'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'
    ancil_path = filepath
    lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'



def load_AWS(station):
	## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
	## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    if host == 'jasmin':
		os.chdir(filepath)
		for file in os.listdir(filepath):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values=-9999, header=0)
    elif host == 'bsl':
		os.chdir('/data/clivarm/wip/ellgil82/AWS/')
		for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
			if fnmatch.fnmatch(file, '%(station)s*' % locals()):
				AWS_srs = pd.read_csv(str(file), na_values = -9999, header = 0)
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    try:
        AWS_srs['time'] = 24.*(AWS_srs['Time'] - AWS_srs['day'])
        time_list = []
        for i in AWS_srs['time']:
            hrs = int(i)                 # will now be 1 (hour)
            mins = int((i-hrs)*60)       # will now be 4 minutes
            secs = int(0 - hrs*60*60 + mins*60) # will now be 30
            j = datetime.time(hour = hrs, minute=mins)
            time_list.append(j)
        AWS_srs['Time'] = time_list
    except TypeError:
        print('Got time already m9')
        AWS_srs['Time'] = pd.to_datetime(AWS_srs['Time'], format='%H:%M:%S').dt.time
    print '\nconverting times...'
    # Convert times so that they can be plotted
    AWS_srs['datetime'] = AWS_srs.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    try:
        AWS_srs['E'] = AWS_srs['LWnet_corr'].values + AWS_srs['SWnet_corr'].values + AWS_srs['Hlat'].values + AWS_srs['Hsen'].values - AWS_srs['Gs'].values
    except:
        print('No full SEB \'ere pal...')
    AWS_srs['WD'][AWS_srs['WD'] < 0.] = np.nan
    AWS_srs['FF_10m'][AWS_srs['FF_10m'] < 0.] = np.nan
    AWS_srs['WD'] = AWS_srs['WD'].interpolate() # interpolate missing values
    AWS_srs['FF_10m'] = AWS_srs['FF_10m'].interpolate()
    AWS_srs['WD'][AWS_srs['WD'] == 0.] = np.nan
    if station == 'AWS14_SEB_2009-2017_norp':
        AWS_srs = AWS_srs.tail(1).append(AWS_srs.iloc[:-1])
    # Calculate months
    months = [g for n, g in AWS_srs.groupby(pd.TimeGrouper('M'))]
    DJF = pd.concat((months[11], months[0], months[1]), axis=0)
    MAM = pd.concat((months[2], months[3], months[4]), axis=0)
    JJA = pd.concat((months[5], months[6], months[7]), axis=0)
    SON = pd.concat((months[8], months[9], months[10]), axis=0)
    if host == 'jasmin':
        os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/hindcast/output/')
    return AWS_srs, DJF, MAM, JJA, SON



AWS14_SEB, DJF_14, MAM_14, JJA_14, SON_14 = load_AWS('AWS14_SEB_2009-2017_norp.csv')
AWS15_SEB, DJF_15, MAM_15, JJA_15, SON_15 = load_AWS('AWS15_hourly_2009-2014.csv')
AWS17_SEB, DJF_17, MAM_17, JJA_17, SON_17 = load_AWS('AWS17_SEB_2011-2015_norp.csv')
AWS18_SEB, DJF_18, MAM_18, JJA_18, SON_18 = load_AWS('AWS18_SEB_2014-2017_norp.csv')

AWS_dict = {'AWS14': AWS14_SEB, 'AWS15': AWS15_SEB, 'AWS17': AWS17_SEB, 'AWS18': AWS18_SEB}


def load_time_srs():
    melt = iris.load_cube('1998-2017_land_snow_melt_amnt_daymn.nc', 'Snowmelt')
    #melt.coord('time').convert_units('days since 1970-01-01 00:00:00', calendar = 'standard')
    Ts = iris.load_cube('1998-2017_Ts_monmean.nc', 'surface_temperature')
    Ts_max = iris.load_cube('1998-2017_Ts_monmax.nc', 'surface_temperature')
    Ts_min = iris.load_cube('1998-2017_Ts_monmin.nc', 'surface_temperature')
    Tair = iris.load_cube('1998-2017_Tair_1p5m_monmean.nc', 'air_temperature')
    Tair_max = iris.load_cube('1998-2017_Tair_1p5m_monmax.nc', 'air_temperature')
    Tair_min = iris.load_cube('1998-2017_Tair_1p5m_monmin.nc', 'air_temperature')
    for i in [Ts, Ts_max, Ts_min, Tair, Tair_min, Tair_max]:
        i.convert_units('celsius')
    RH = iris.load_cube('1998-2017_RH_1p5m_monmean.nc', 'relative_humidity')
    RH_min = iris.load_cube('1998-2017_RH_1p5m_monmin.nc', 'relative_humidity')
    RH_max = iris.load_cube('1998-2017_RH_1p5m_monmax.nc', 'relative_humidity')
    FF_10m = iris.load_cube('1998-2017_FF_10m_monmean.nc', 'wind_speed')
    FF_min = iris.load_cube('1998-2017_FF_10m_monmin.nc', 'wind_speed')
    FF_max = iris.load_cube('1998-2017_FF_10m_monmax.nc', 'wind_speed')
    FF_10m = FF_10m[:,:, :220, :220]
    FF_min = FF_min[:, :, :220, :220]
    FF_max = FF_max[:, :, :220, :220]
    # Rotate data onto standard lat/lon grid
    for i in [melt, Ts, Tair, RH, FF_10m]:
        real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
    var_dict = {'melt': melt[:,0,:,:], 'Ts': Ts[:,0,:,:], 'Ts_max': Ts_max[:,0,:,:], 'Ts_min': Ts_min[:,0,:,:], 'FF_10m': FF_10m[:,0,:,:],
                'FF_10m_min': FF_min[:,0,:,:], 'FF_10m_max': FF_max[:,0,:,:],'RH': RH[:,0,:,:], 'RH_min': RH_min[:,0,:,:], 'RH_max': RH_max[:,0,:,:],
                'Time_srs': Ts.coord('time').points, 'Tair': Tair[:,0,:,:], 'Tair_max': Tair[:,0,:,:], 'Tair_min': Tair[:,0,:,:], 'lat': real_lat, 'lon': real_lon}
    return var_dict


#all_vars = load_time_srs()

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


def Larsen_melt_srs():
    # Create Larsen mask
    orog = iris.load_cube(ancil_path + 'orog.nc', 'surface_altitude')
    orog = orog[0, 0, :, :]
    LSM = iris.load_cube(ancil_path + 'new_mask.nc', lsm_name)
    lsm = LSM[0, 0, :, :]
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = lsm.data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[orog.data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    all_vars['melt'].data = np.ma.masked_array(all_vars['melt'].data, mask=np.broadcast_to(Larsen_mask, all_vars['melt'].shape))
    mean_integrated_melt = all_vars['melt'].data.mean(axis=(1, 2))
    fig, ax = plt.subplots(1,1, figsize = (18,7))
    ax2 = ax.twinx()
    for axs in [ax, ax2]:
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=32, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.plot(all_vars['melt'].coord('time').points, mean_integrated_melt, color = '#1f78b4', zorder = 2)
    ax2.plot(all_vars['melt'].coord('time').points, np.cumsum(mean_integrated_melt), color='#f68080', lw = 5, zorder = 4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_color('dimgrey')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #days = mdates.MonthLocator(interval=1)
    #dayfmt = mdates.DateFormatter('%b %Y')
    ax2.set_ylabel('Cumulative \nmeltwater \nproduction \n (kg m$^{-2}$)', rotation=0, fontsize=32, labelpad=100, color='dimgrey')
    ax.set_ylabel('Daily mean \nice shelf-\nintegrated \nmeltwater \nproduction \n(kg m$^{-2}$)', rotation=0, fontsize=32, labelpad=100, color='dimgrey')
    ax.set_xlim(all_vars['melt'].coord('time').points[0], all_vars['melt'].coord('time').points[-1])
    #ax.xaxis.set_major_formatter(dayfmt)
    ax.yaxis.set_label_coords(-0.25, 0.25)
    ax2.yaxis.set_label_coords(1.23, 0.75)
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 5 != 0]
    # Set limits
    ax2.set_ylim(0,3.0)
    ax.set_ylim(0, 0.015)
    ax2.set_yticks([0,1.0, 2.0, 3.0])
    ax.set_yticks([0,.0050, .010, .015 ])
    plt.subplots_adjust(left = 0.22, right = 0.8)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_melt_amount_Larsen_C_integrated.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_melt_amount_Larsen_C_integrated.eps')
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_melt_amount_Larsen_C_integrated.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_melt_amount_Larsen_C_integrated.eps')
    plt.show()

#Larsen_melt_srs()

def surf_plot():
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'AWS14': '#33a02c', 'AWS15': '#f68080', 'AWS17': '#1f78b4', 'AWS18': '#FF6522', lat_index14: '#33a02c', lat_index15: '#f68080', lat_index17: '#1f78b4', lat_index18: '#FF6522'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    #days = mdates.DayLocator(interval=1)
    #dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    for j in ['RH', 'FF_10m', 'Tair', 'Ts']:
        limits = {'RH': (0,100), 'FF_10m': (0,25), 'Tair': (-30, 5), 'Ts': (-30, 5)}
        titles = {'RH': 'Relative \nhumidity (%)',
                  'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                  'Tair': '2 m air \ntemperature ($^{\circ}$C)',
                  'Ts': 'Surface \ntemperature ($^{\circ}$C)'}
        for AWS in ['AWS14', 'AWS15', 'AWS17', 'AWS18']:
            ax[plot].plot(all_vars['Time_srs'], all_vars[j][:, lat_dict[AWS], lon_dict[AWS]].data, color = col_dict[AWS], linewidth = 2.5, label = 'MetUM output at ' + AWS)
         #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
        ax[plot].set_xlim(all_vars['Time_srs'][1], all_vars['Time_srs'][-1])
        ax[plot].set_ylim(limits[j])#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color = 'dimgrey', labelpad = 80)
        ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        lab = ax[plot].text(0.08, 0.85, zorder = 100, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        plt.setp(axs.get_yticklabels()[-2], visible=False)
        #axs.xaxis.set_major_formatter(dayfmt)
        #plt.setp(axs.get_xticklabels()[])
    for axs in [ax[0], ax[1]]:
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(all_vars['Time_srs'][1], all_vars['Time_srs'][-1])
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    # Legend
    lgd = ax[1].legend(bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_surface_met_modelled.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_surface_met_modelled.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_surface_met_modelled.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_surface_met_modelled.eps')
    plt.show()

#surf_plot()

def minmax_plot(AWS):
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'AWS14': '#33a02c', 'AWS15': '#f68080', 'AWS17': '#1f78b4', 'AWS18': '#FF6522', lat_index14: '#33a02c', lat_index15: '#f68080', lat_index17: '#1f78b4', lat_index18: '#FF6522'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    #days = mdates.DayLocator(interval=1)
    #dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    for j in ['RH', 'FF_10m', 'Tair', 'Ts']:
        limits = {'RH': (0,100), 'FF_10m': (0,25), 'Tair': (-35, 5), 'Ts': (-35, 5)}
        titles = {'RH': 'Relative \nhumidity (%)',
                  'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                  'Tair': '2 m air \ntemperature ($^{\circ}$C)',
                  'Ts': 'Surface \ntemperature ($^{\circ}$C)'}
        ax[plot].plot(all_vars['Time_srs'], all_vars[j][:, lat_dict[AWS], lon_dict[AWS]].data, color = col_dict[AWS], linewidth = 2.5, label = 'MetUM output at ' + AWS)
        ax[plot].plot(all_vars['Time_srs'], all_vars[j + '_max'][:, lat_dict[AWS], lon_dict[AWS]].data, color = col_dict[AWS], linewidth = 1, linestyle = ':', label = 'MetUM output at ' + AWS)
        ax[plot].plot(all_vars['Time_srs'], all_vars[j + '_min'][:, lat_dict[AWS], lon_dict[AWS]].data,color=col_dict[AWS], linewidth=1, linestyle=':', label='MetUM output at ' + AWS)
        ax[plot].fill_between(all_vars['Time_srs'], all_vars[j + '_min'][:, lat_dict[AWS], lon_dict[AWS]].data, all_vars[j + '_max'][:, lat_dict[AWS], lon_dict[AWS]].data, color = col_dict[AWS], alpha = 0.4)
         #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
        ax[plot].set_xlim(all_vars['Time_srs'][1], all_vars['Time_srs'][-1])
        ax[plot].set_ylim(limits[j])#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color = 'dimgrey', labelpad = 80)
        ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        lab = ax[plot].text(0.08, 0.85, zorder = 100, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On = False)
        axs.spines['left'].set_visible(False)
    #for axs in [ax[2], ax[3]]:
        #axs.xaxis.set_major_formatter(dayfmt)
        #plt.setp(axs.get_xticklabels()[])
    for axs in [ax[0], ax[1]]:
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(all_vars['Time_srs'][1], all_vars['Time_srs'][-1])
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    # Legend
    lgd = ax[1].legend(bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_surface_met_modelled.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_surface_met_modelled.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_surface_met_modelled_minmax_' + AWS + '.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_surface_met_modelled_minmax_' + AWS + '.eps')
    plt.show()

minmax_plot('AWS14')
minmax_plot('AWS15')
minmax_plot('AWS17')
minmax_plot('AWS18')

## Question: are 'inlet stations' significantly different from 'ice shelf' stations?

ice_shelf_melt = np.mean(melt_full_srs[:, lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2))+ np.mean(melt_full_srs[:, lat_index15-1:lat_index15+1, lon_index15-1:lon_index15+1].data, axis = (1,2))/2.
inlet_melt = np.mean(melt_full_srs[:, lat_index17-1:lat_index17+1, lon_index17-1:lon_index17+1].data, axis = (1,2))+  np.mean(melt_full_srs[:, lat_index18-1:lat_index18+1, lon_index18-1:lon_index18+1].data, axis = (1,2)) / 2.


t,p = scipy.stats.ttest_ind(ice_shelf_melt, inlet_melt)

FF_full_srs = iris.load_cube('1998-2017_FF_10m.nc')
FF_full_srs = FF_full_srs[:,0,:220,:]

ice_shelf_FF = np.mean(all_vars['FF_10m'][:, lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2))+ np.mean(all_vars['FF_10m'][:, lat_index15-1:lat_index15+1, lon_index15-1:lon_index15+1].data, axis = (1,2))/2.
inlet_FF = np.mean(all_vars['FF_10m'][:, lat_index17-1:lat_index17+1, lon_index17-1:lon_index17+1].data, axis = (1,2))+  np.mean(all_vars['FF_10m'][:, lat_index18-1:lat_index18+1, lon_index18-1:lon_index18+1].data, axis = (1,2)) / 2.

def calc_total_melt():
    # Create Larsen mask
    orog = iris.load_cube(ancil_path + 'orog.nc', 'surface_altitude')
    orog = orog[0, 0, :, :]
    LSM = iris.load_cube(ancil_path + 'new_mask.nc', lsm_name)
    lsm = LSM[0, 0, :, :]
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = lsm.data[40:135, 90:160]
    Larsen_mask[40:135, 90:160] = lsm_subset
    Larsen_mask[orog.data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    # Mask data
    all_vars['melt'].data = np.ma.masked_array(all_vars['melt'].data, mask=np.broadcast_to(Larsen_mask, all_vars['melt'].shape))
    # Calculate melt total over all years
    melt_sum = all_vars['melt'].collapsed('time', iris.analysis.SUM) # total melt at each grid point over 20 years (kg m-2)

    totm = all_vars['melt'].collapsed('latitude', iris.analysis.SUM)

    #totm = melt_sum.collapsed('latitude', iris.analysis.SUM)
    totm = totm.collapsed('longitude', iris.analysis.SUM)
    tot_annual_melt = totm.data/20.
    return tot_annual_melt



## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

from windrose import WindroseAxes


def wind_rose(station, AWS_var, model_var):
    fig = plt.figure(figsize = (16,8))
    rect = [0.05, 0.1, 0.45, 0.6]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    # define data limits
    max_mod = max(np.mean(vars_yr['FF_10m'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)))
    max_obs = max(AWS_var['FF'])
    wa.set_title('Observed', fontsize = 28, color = 'dimgrey', pad = 50)
    wa.axes.spines['polar'].set_visible(False)
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    wa.bar(AWS_var['WD'], AWS_var['FF'], bins = np.arange(0, max(max_mod, max_obs),4), cmap = plt.get_cmap('viridis'), normed = True,opening=0.8, edgecolor='white')
    wa.set_yticklabels([])
    rect = [0.55, 0.1, 0.45, 0.6]
    wa = WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.set_title('Modelled', fontsize=28, color='dimgrey', pad = 50)
    wa.bar(np.mean(vars_yr['WD'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)),
           np.mean(vars_yr['FF_10m'][:,lat_index14-1:lat_index14+1, lon_index14-1:lon_index14+1].data, axis = (1,2)),
           bins = np.arange(0, max(max_mod, max_obs),4), cmap = plt.get_cmap('viridis'),  normed = True, opening=0.8, edgecolor='white')
    lgd = wa.set_legend( bbox_to_anchor=(-0.5, 0.9))
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey', fontsize = 18)
    lgd.get_frame().set_linewidth(0.0)
    wa.axes.spines['polar'].set_visible(False)
    wa.set_yticklabels([])
    wa.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    plt.savefig('/users/ellgil82/figures/Hindcast/validation/wind_rose_' + station + '_' + year + '.png')
    plt.savefig('/users/ellgil82/figures/Hindcast/validation/wind_rose_' + station + '_' + year + '.eps')
    plt.show()


def all_sta_wind_rose():
    fig = plt.figure(figsize=(16, 16))
    rect = [[0.05, 0.55, 0.35, 0.35], [0.55, 0.55, 0.35, 0.35], [0.05, 0.05, 0.35, 0.35], [0.55, 0.05,0.35, 0.35]]
    plot = 0
    for station in station_dict.keys():
        wa = WindroseAxes(fig, rect[plot])
        fig.add_axes(wa)
        ANN, DJF, MAM, JJA, SON = load_AWS(station)
        # define data limits
        max_mod = max(ANN['FF_10m'])
        wa.set_title(station_dict[station], fontsize=28, color='dimgrey', pad=50)
        wa.axes.spines['polar'].set_visible(False)
        wa.tick_params(axis='both', which='both', labelsize=20, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        wa.bar(ANN['WD'], ANN['FF_10m'], bins=np.arange(0, 20, 4), cmap=plt.get_cmap('viridis'), normed=True, opening=0.8, edgecolor='white')
        wa.set_yticks([5,10,15])
        wa.set_yticklabels([ '', '10%', '15%'])
        plot = plot + 1
    lgd = wa.set_legend(bbox_to_anchor=(-0.45, 1.))
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey', fontsize=24)
    lgd.get_frame().set_linewidth(0.0)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/Validation/wind_rose_all_stations_all_years.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/Validation/wind_rose_all_stations_all_years.eps')
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/wind_rose_all_stations_all_years.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/wind_rose_all_stations_all_years.eps')
    plt.show()

all_sta_wind_rose()


'''
# Create pandas dataframe with each file loaded in as a time series and then added as a column (would only work for specific locations/whole ice shelf mean)
ice_shelf = pd.DataFrame()
AWS14 = pd.DataFrame()
AWS15 = pd.DataFrame()
AWS17 = pd.DataFrame()
AWS18 = pd.DataFrame()
for i in ['Ts', 'Tair', 'melt_amnt', 'melt_flux']:
    ice_shelf_masked_srs = np.ma.masked_where(var_list['orog'].data >= 100. and var_list['lsm'].data == 0, var_list[i].data, copy = True)
    ice_shelf_srs = np.mean(ice_shelf_masked_srs, axis = (1,2))
    AWS14_srs = np.mean(i[:,lat14_index, lon14_index].data, axis = (1,2))
    AWS15_srs = np.mean(i[:,lat15_index, lon15_index].data, axis = (1,2))
    AWS17_srs = np.mean(i[:,lat17_index, lon17_index].data, axis = (1,2))
    AWS18_srs = np.mean(i[:,lat18_index, lon18_index].data, axis = (1,2))
    ice_shelf[i] = ice_shelf_srs
    AWS14[i] = AWS14_srs
    AWS15[i] = AWS15_srs
    AWS17[i] = AWS17_srs
    AWS18[i] = AWS18_srs

'''
