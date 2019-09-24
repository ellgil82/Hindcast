
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
import cf_units
from divg_temp_colourmap import shiftedColorMap
import time
from sklearn.metrics import mean_squared_error
import datetime
import cftime
import metpy
import metpy.calc
import glob
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

def load_var(seas):
    # Load variables
    os.chdir(filepath)
    LWnet = iris.load_cube(filepath + seas + '_diurnal_surface_LW_net.nc', 'surface_net_downward_longwave_flux')
    SWnet = iris.load_cube(filepath + seas + '_diurnal_surface_SW_net.nc', 'Net short wave radiation flux')
    LWdown = iris.load_cube(filepath + seas + '_diurnal_surface_LW_down.nc', 'IR down')
    SWdown = iris.load_cube(filepath + seas + '_diurnal_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
    HL = iris.load_cube(filepath + seas + '_diurnal_latent_heat.nc', 'Latent heat flux')
    HS = iris.load_cube(filepath + seas + '_diurnal_sensible_heat.nc', 'surface_upward_sensible_heat_flux')
    melt_flux = iris.load_cube(filepath + seas + '_diurnal_land_snow_melt_flux.nc', 'Snow melt heating flux')
    if host == 'bsl':
        try:
            LSM = iris.load_cube(filepath + 'new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')
            orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
            orog = orog[0, 0, :, :]
            LSM = LSM[0, 0, :, :]
        except iris.exceptions.ConstraintMismatchError:
            print('Files not found')
    elif host == 'jasmin':
        try:
            LSM = iris.load_cube(filepath + 'new_mask.nc', 'land_binary_mask')
            orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
            orog = orog[0, 0, :, :]
            LSM = LSM[0, 0, :, :]
        except iris.exceptions.ConstraintMismatchError:
            print('Files not found')
    # Create standardised time units
    t = [cftime.datetime(0, 0, 0, 0), cftime.datetime(3, 0, 0, 0), cftime.datetime(6, 0, 0, 0), cftime.datetime(9, 0, 0, 0),
         cftime.datetime(12, 0, 0, 0), cftime.datetime(15, 0, 0, 0), cftime.datetime(18, 0, 0, 0), cftime.datetime(21, 0, 0, 0)]
    t_num = [0,3,6,9,12,15,18,21]
    new_time = iris.coords.AuxCoord(t, long_name='time', standard_name='time', units=cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='standard'))
    T_dim = iris.coords.DimCoord(t_num, long_name='time', standard_name='time', units=cf_units.Unit('hours since 1970-01-01 00:00:00', calendar='standard'))
    # Rotate data onto standard lat/lon grid and update times
    for i in [orog, LWnet, SWnet, HL, HS, SWdown, LWdown, melt_flux]:
        real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
    for i in [LWnet, SWnet, HL, HS, SWdown, LWdown, melt_flux]:
        try:
            i.remove_coord('time')
        except iris.exceptions.CoordinateNotFoundError:
            i.remove_coord('t')
        i.add_aux_coord(new_time, 0)
        i.attributes = {'north_pole': [296.  ,  22.99], 'name': 'solar', 'title': 'Net short wave radiation flux', 'CDO': 'Climate Data Operators version 1.9.5 (http://mpimet.mpg.de/cdo)', 'CDI': 'Climate Data Interface version 1.9.5 (http://mpimet.mpg.de/cdi)', 'Conventions': 'CF-1.6', 'source': 'Unified Model Output (Vn11.1):', 'time': '12:00', 'date': '31/12/97'}
    # Calculate Etot
    Etot = iris.cube.Cube(data = LWnet.data + SWnet.data - HL.data - HS.data,  long_name = 'Total energy flux', var_name = 'Etot', units = SWnet.units)
    for n in range(3):
        Etot.add_dim_coord(SWnet.dim_coords[n],n+1)
    Etot.add_aux_coord(SWnet.aux_coords[0], 0)
    # Flip direction of turbulent fluxes to match convention (positive towards surface)
    for turb in [HS, HL]:
        turb.data = 0 - turb.data
    seas_SEB = {'lon': real_lon, 'lat': real_lat, 'seas': seas, 'LWnet': LWnet[:,0,:,:],  'SWnet': SWnet[:,0,:,:],
                'LWdown': LWdown[:,0,:,:],  'SWdown': SWdown[:,0,:,:],  'HL': HL[:,0,:,:],  'HS': HS[:,0,:,:],
                'Etot': Etot[:,0,:,:], 'melt': melt_flux[:,0,:,:], 'Time_srs': t_num}
    return seas_SEB

seas_var = load_var('DJF')

lon_index14, lat_index14, = find_gridbox(-67.01, -61.03, seas_var['lat'], seas_var['lon'])
lon_index15, lat_index15, = find_gridbox(-67.34, -62.09, seas_var['lat'], seas_var['lon'])
lon_index17, lat_index17, = find_gridbox(-65.93, -61.85, seas_var['lat'], seas_var['lon'])
lon_index18, lat_index18, = find_gridbox(-66.48272, -63.37105, seas_var['lat'], seas_var['lon'])

lat_dict = {'AWS14': lat_index14,
            'AWS15': lat_index15,
            'AWS17': lat_index17,
            'AWS18': lat_index18}

lon_dict = {'AWS14': lon_index14,
            'AWS15': lon_index15,
            'AWS17': lon_index17,
            'AWS18': lon_index18}


def total_SEB_model(seas_list, loc):
    for seas in seas_list:
        DJF = load_var(seas)
        fig, ax = plt.subplots(1,1, figsize = (18,8),frameon= False)
        days = mdates.DayLocator(interval=1)
        dayfmt = mdates.DateFormatter('%b %d')
        plt.setp(ax.spines.values(), linewidth=2, color = 'dimgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(-50, 150)
        ax.set_xlim(DJF['Time_srs'][0], DJF['Time_srs'][-1])
        ax.plot(DJF['Time_srs'], DJF['SWnet'][:,lon_dict[loc], lat_dict[loc]].data, color = '#6fb0d2', lw = 2.5, label = 'Net shortwave flux')
        ax.plot(DJF['Time_srs'], DJF['LWnet'][:,lon_dict[loc], lat_dict[loc]].data, color = '#86ad63', lw = 2.5, label = 'Net longwave flux')
        ax.plot(DJF['Time_srs'], DJF['HS'][:,lon_dict[loc], lat_dict[loc]].data, color = '#1f78b4', lw = 2.5, label = 'Sensible heat flux')
        ax.plot(DJF['Time_srs'], DJF['HL'][:,lon_dict[loc], lat_dict[loc]].data, color = '#33a02c', lw = 2.5, label = 'Latent heat flux')
        ax.plot(DJF['Time_srs'], DJF['melt'][:,lon_dict[loc], lat_dict[loc]].data, color = '#f68080', lw = 2.5, label = 'Melt flux')
        lgd = plt.legend(fontsize = 18, frameon = False)
        for ln in lgd.get_texts():
            plt.setp(ln, color = 'dimgrey')
        #ax.xaxis.set_major_formatter(dayfmt)
        #ax.text(x=SEB_1p5['Time_srs'][6], y=350, s='b', fontsize=32, fontweight='bold', color='dimgrey')
        ax.tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On=False, labelcolor = 'dimgrey', pad = 10)
        ax.axhline(y=0, xmin = 0, xmax = 1, linestyle = '--', linewidth = 1)
        ax.set_ylabel('Energy flux \n (W m$^{-2}$)',  rotation = 0, fontsize = 28, labelpad = 70, color='dimgrey')
        plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95)
        [l.set_visible(False) for (w,l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w,l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_SEB_' + seas + '_' + loc + '.eps', transparent=True )
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_SEB_' + seas + '_' + loc + '.png', transparent=True )
        plt.show()

for station in ['AWS14', 'AWS15', 'AWS17', 'AWS18']:
    total_SEB_model(['DJF', 'MAM', 'JJA', 'SON'], loc = station)

DJF = load_var('DJF')
JJA = load_var('JJA')



def SEB_subplot():
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    colour_dict = {'SWdown': '#6fb0d2', 'SWnet': '#6fb0d2', 'SWup': '#6fb0d2', 'LWdown': '#86ad63', 'LWnet': '#86ad63', 'LWup': '#86ad63','HL': '#33a02c', 'HS': '#1f78b4', 'melt': '#f68080'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    UTC_DJF_inlet = pd.DataFrame()
    UTC_JJA_inlet = pd.DataFrame()
    UTC_DJF_shelf = pd.DataFrame()
    UTC_JJA_shelf = pd.DataFrame()
    for j in ['SWnet', 'LWnet', 'HS', 'HL', 'melt']:
        ice_shelf_mn = ((DJF[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + DJF[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((DJF[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + DJF[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_DJF_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_DJF_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf_mn = ((JJA[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + JJA[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((JJA[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + JJA[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_JJA_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_JJA_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf = ax[0].plot(UTC_DJF_shelf[j], color=colour_dict[j], lw=2.5, label = j)
        inlet = ax[1].plot(UTC_DJF_inlet[j], color=colour_dict[j], lw=2.5, label = j)
        ax[2].plot(UTC_JJA_inlet[j], color=colour_dict[j], lw=2.5, label = j)
        ax[3].plot(UTC_JJA_shelf[j], color=colour_dict[j], lw=2.5, label = j)
     #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
    for i in range(4):
        #ax[i].set_xlim(DJF['Time_srs'][1], DJF['Time_srs'][-1])
        ax[i].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        ax[i].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
        ax[i].set_ylim(-60, 120)#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[i].set_xlim(0,8)
        lab = ax[i].text(0.08, 0.85, zorder = 100, transform = ax[i].transAxes, s=lab_dict[i], fontsize=32, fontweight='bold', color='dimgrey')
        ax[i].spines['top'].set_visible(False)
        plt.setp(ax[i].spines.values(), linewidth=2, color='dimgrey', )
        ax[i].tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick2On = False, tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        plt.xticks(range(8), ('00:00', '03:00', '06:00','09:00', '12:00', '15:00', '18:00', '21:00', '00:00'))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 3 != 0]
    # Legend
    lgd = ax[3].legend(bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    ax[0].set_title('Ice shelf stations', fontsize = 30, color = 'dimgrey')
    ax[1].set_title('Inlet stations', fontsize = 30, color = 'dimgrey')
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB.eps')
    plt.show()

SEB_subplot()