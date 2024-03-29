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
import copy
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
    albedo = SWdown.data / (SWnet.data - SWdown.data)
    albedo = iris.cube.Cube(data=albedo, long_name = 'Surface albedo', var_name = 'albedo',)
    for n in range(3):
        Etot.add_dim_coord(SWnet.dim_coords[n],n+1)
        albedo.add_dim_coord(SWnet.dim_coords[n],n+1)
    Etot.add_aux_coord(SWnet.aux_coords[0], 0)
    albedo.add_aux_coord(SWnet.aux_coords[0], 0)
    # Flip direction of turbulent fluxes to match convention (positive towards surface)
    HS = iris.analysis.maths.multiply(HS, -1.)
    HL = iris.analysis.maths.multiply(HL, -1.)
    seas_SEB = {'lon': real_lon, 'lat': real_lat, 'seas': seas, 'LWnet': LWnet[:,0,:,:],  'SWnet': SWnet[:,0,:,:],
                'LWdown': LWdown[:,0,:,:],  'SWdown': SWdown[:,0,:,:],  'HL': HL[:,0,:,:],  'HS': HS[:,0,:,:],
                'Etot': Etot[:,0,:,:], 'melt': melt_flux[:,0,:,:], 'Time_srs': t_num, 'albedo': albedo[:,0,:,:]}
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
        ax.set_ylim(-100, 100)
        ax.set_xlim(DJF['Time_srs'][0], DJF['Time_srs'][-1])
        ax.plot(DJF['Time_srs'], DJF['SWnet'][:,lon_dict[loc], lat_dict[loc]].data, marker = '+', color = '#6fb0d2', lw = 4, label = 'Net shortwave flux')
        ax.plot(DJF['Time_srs'], DJF['LWnet'][:,lon_dict[loc], lat_dict[loc]].data, marker = 'X', color = '#86ad63', lw = 4, label = 'Net longwave flux')
        ax.plot(DJF['Time_srs'], DJF['HS'][:,lon_dict[loc], lat_dict[loc]].data, marker = 'O', color = '#1f78b4', lw = 4, label = 'Sensible heat flux')
        ax.plot(DJF['Time_srs'], DJF['HL'][:,lon_dict[loc], lat_dict[loc]].data, marker = '*', color = '#33a02c', lw = 4, label = 'Latent heat flux')
        ax.plot(DJF['Time_srs'], DJF['melt'][:,lon_dict[loc], lat_dict[loc]].data, color = '#f68080', lw = 4, label = 'Melt flux')
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
MAM = load_var('MAM')
SON = load_var('SON')

def SEB_subplot():
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(24, 12))
    ax = ax.flatten()
    colour_dict = {'SWdown': ('#6fb0d2', 'o'), 'SWnet': ('#b733ff', 'o'), 'SWup': ('#6fb0d2', 'o'), 'LWdown': ('#86ad63','X'), 'LWnet': ('#de2d26','X'), 'LWup': ('#86ad63','X'),'HL': ('#33aeff', '*'), 'HS': ('#ff6500', '^'), 'Etot': ('#222222', 's'), 'melt': ('#f68080', 'P')}
    leg_dict = {'SWnet': 'SW$_{net}$', 'LWnet': 'LW$_{net}$', 'HL': 'H$_{L}$', 'HS': 'H$_{S}$', 'Etot': 'E$_{tot}$', 'melt': 'E$_{melt}$'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    UTC_DJF_inlet = pd.DataFrame()
    UTC_JJA_inlet = pd.DataFrame()
    UTC_DJF_shelf = pd.DataFrame()
    UTC_JJA_shelf = pd.DataFrame()
    for j in ['SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'melt']:
        ice_shelf_mn = ((DJF[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + DJF[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((DJF[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + DJF[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_DJF_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_DJF_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf_mn = ((JJA[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + JJA[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((JJA[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + JJA[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_JJA_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_JJA_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf = ax[0].plot(UTC_DJF_shelf[j], color=colour_dict[j][0], marker = colour_dict[j][1],  markersize = 15, lw=4, label = leg_dict[j])
        inlet = ax[1].plot(UTC_DJF_inlet[j], color=colour_dict[j][0],marker = colour_dict[j][1], markersize = 15, lw=4, label = leg_dict[j])
        ax[2].plot(UTC_JJA_inlet[j], color=colour_dict[j][0], marker = colour_dict[j][1], lw=4, markersize = 15, label = leg_dict[j])
        ax[3].plot(UTC_JJA_shelf[j], color=colour_dict[j][0], marker = colour_dict[j][1], lw=4, markersize = 15, label = leg_dict[j])
     #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
    for i in range(4):
        #ax[i].set_xlim(DJF['Time_srs'][1], DJF['Time_srs'][-1])
        ax[i].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        ax[i].set_ylim(-60, 120)#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[i].set_xlim(0,8.01)
        lab = ax[i].text(0.08, 0.85, zorder = 100, transform = ax[i].transAxes, s=lab_dict[i], fontsize=32, fontweight='bold', color='dimgrey')
        ax[i].spines['top'].set_visible(False)
        plt.setp(ax[i].spines.values(), linewidth=2, color='dimgrey', )
        ax[i].tick_params(axis='both', which='major', length = 8, width = 2, color = 'dimgrey', labelsize=24, tick1On=True, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[i].tick_params(axis='both', which='minor', length=0, width=0, color='dimgrey', labelsize=24, tick1On=True,
                          tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (w, l) in enumerate(ax[i].xaxis.get_major_ticks()) if w % 2 != 0]
        #ax[i].xaxis.get_major_ticks()[-2].set_visible(False)
    ax[0].yaxis.set_label_coords(0, 1)
    ax[1].yaxis.set_label_coords(1, 1)
    ax[0].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    ax[1].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    for axs in [ax[0], ax[2]]:
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_ticks_position('right')
        #axs.tick_params(axis='y', tick2On = False, tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        axs.yaxis.set_label(' ')
        plt.xticks(range(9), ('00:00', '03:00', '06:00','09:00', '12:00', '15:00', '18:00', '21:00', '00:00', '00:00'))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 4 != 0]
    # Legend
    lgd = ax[3].legend(bbox_to_anchor=(0.3, 1), ncol = 2, loc=2, fontsize=20)
    ax[0].set_title('Ice shelf stations', fontsize = 30, color = 'dimgrey')
    ax[0].text(-0.4, 0.55, zorder=100, transform=ax[0].transAxes, s='DJF', fontsize=32, fontweight='bold',color='dimgrey')
    ax[2].text(-0.4, 0.55, zorder=100, transform=ax[2].transAxes, s='JJA', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_title('Inlet stations', fontsize = 30, color = 'dimgrey')
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.15, hspace = 0.05, top = 0.92, right = 0.85, left = 0.16, bottom = 0.08)
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

def SEB_caseplot(case):
    case_var = load_var(case)
    fig, ax = plt.subplots(1,2,  figsize=(24, 6))
    ax = ax.flatten()
    colour_dict = {'SWdown': ('#6fb0d2', 'o'), 'SWnet': ('#b733ff', 'o'), 'SWup': ('#6fb0d2', 'o'), 'LWdown': ('#86ad63','X'), 'LWnet': ('#de2d26','X'), 'LWup': ('#86ad63','X'),'HL': ('#33aeff', '*'), 'HS': ('#ff6500', '^'), 'Etot': ('#222222', 's'), 'melt': ('#f68080', 'P')}
    leg_dict = {'SWnet': 'SW$_{net}$', 'LWnet': 'LW$_{net}$', 'HL': 'H$_{L}$', 'HS': 'H$_{S}$', 'Etot': 'E$_{tot}$', 'melt': 'E$_{melt}$'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    UTC_case_inlet = pd.DataFrame()
    UTC_case_shelf = pd.DataFrame()
    for j in ['SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'melt']:
        ice_shelf_mn = ((case_var[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + case_var[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((case_var[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + case_var[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_case_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_case_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf = ax[0].plot(UTC_case_shelf[j], color=colour_dict[j][0], marker = colour_dict[j][1],  markersize = 15, lw=4, label = leg_dict[j])
        inlet = ax[1].plot(UTC_case_inlet[j], color=colour_dict[j][0], marker = colour_dict[j][1],  markersize = 15, lw=4, label = leg_dict[j])
     #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
    for i in range(2):
        #ax[i].set_xlim(DJF['Time_srs'][1], DJF['Time_srs'][-1])
        ax[i].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        ax[i].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
        ax[i].set_ylim(-60, 120)#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[i].set_xlim(0,8.01)
        lab = ax[i].text(0.08, 0.85, zorder = 100, transform = ax[i].transAxes, s=lab_dict[i], fontsize=32, fontweight='bold', color='dimgrey')
        ax[i].spines['top'].set_visible(False)
        plt.setp(ax[i].spines.values(), linewidth=2, color='dimgrey', )
        ax[i].tick_params(axis='both', which='major', length = 8, width = 2, color = 'dimgrey', labelsize=24, tick1On=True, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[i].tick_params(axis='both', which='minor', length=0, width=0, color='dimgrey', labelsize=24, tick1On=True,
                          tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (w, l) in enumerate(ax[i].xaxis.get_major_ticks()) if w % 2 != 0]
        #ax[i].xaxis.get_major_ticks()[-2].set_visible(False)
    ax[0].yaxis.set_label_coords(-0.3, 0.5)
    ax[0].spines['right'].set_visible(False)
    ax[1].yaxis.set_label_coords(1.27, 0.5)
    ax[1].yaxis.set_ticks_position('right')
    #axs.tick_params(axis='y', tick2On = False, tick1On = False)
    ax[1].spines['left'].set_visible(False)
    for axs in [ax[0], ax[1]]:
        plt.xticks(range(9), ('00:00', '03:00', '06:00','09:00', '12:00', '15:00', '18:00', '21:00', '00:00', '00:00'))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 4 != 0]
    # Legend
    lgd = ax[1].legend(bbox_to_anchor=(0.3, 1), ncol = 2, loc=2, fontsize=20)
    ax[0].set_title('Ice shelf stations', fontsize = 30, color = 'dimgrey')
    #ax[0].text(-0.65, 0.55, zorder=100, transform=ax[0].transAxes, s='DJF', fontsize=32, fontweight='bold',color='dimgrey')
    #ax[2].text(-0.65, 0.55, zorder=100, transform=ax[2].transAxes, s='JJA', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_title('Inlet stations', fontsize = 30, color = 'dimgrey')
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.15, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.15)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB'+case+'.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB'+case+'.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB'+case+'.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB'+case+'.eps')
    plt.show()

SEB_caseplot('MAM_2016')

#Seasonal cycle plot

def load_ann():
    # Load variables
    os.chdir(filepath)
    LWnet = iris.load_cube(filepath + 'surface_LW_net_climatology.nc', 'surface_net_downward_longwave_flux')
    SWnet = iris.load_cube(filepath + 'surface_SW_net_climatology.nc', 'Net short wave radiation flux')
    LWdown = iris.load_cube(filepath + 'surface_LW_down_climatology.nc', 'IR down')
    SWdown = iris.load_cube(filepath + 'surface_SW_down_climatology.nc', 'surface_downwelling_shortwave_flux_in_air')
    HL = iris.load_cube(filepath + 'latent_heat_climatology.nc', 'Latent heat flux')
    HS = iris.load_cube(filepath + 'sensible_heat_climatology.nc', 'surface_upward_sensible_heat_flux')
    melt_flux = iris.load_cube(filepath + 'land_snow_melt_flux_climatology.nc', 'Snow melt heating flux')
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
    # Rotate data onto standard lat/lon grid and update times
    for i in [orog, LWnet, SWnet, HL, HS, SWdown, LWdown, melt_flux]:
        real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)
    for i in [LWnet, SWnet, HL, HS, SWdown, LWdown, melt_flux]:
        i.attributes = {'north_pole': [296.  ,  22.99], 'name': 'solar', 'title': 'Net short wave radiation flux', 'CDO': 'Climate Data Operators version 1.9.5 (http://mpimet.mpg.de/cdo)', 'CDI': 'Climate Data Interface version 1.9.5 (http://mpimet.mpg.de/cdi)', 'Conventions': 'CF-1.6', 'source': 'Unified Model Output (Vn11.1):', 'time': '12:00', 'date': '31/12/97'}
    # Calculate Etot
    Etot = iris.cube.Cube(data = LWnet.data + SWnet.data - HL.data - HS.data,  long_name = 'Total energy flux', var_name = 'Etot', units = SWnet.units)
    albedo = (SWnet.data - SWdown.data)*-1. / SWdown.data
    albedo[albedo==0] = np.nan
    albedo = iris.cube.Cube(data=albedo, long_name = 'Surface albedo', var_name = 'albedo',)
    for n in range(3):
        Etot.add_dim_coord(SWnet.dim_coords[n],n+1)
        albedo.add_dim_coord(SWnet.dim_coords[n],n+1)
    Etot.add_aux_coord(SWnet.aux_coords[0], 0)
    albedo.add_aux_coord(SWnet.aux_coords[0], 0)
    # Flip direction of turbulent fluxes to match convention (positive towards surface)
    HS = iris.analysis.maths.multiply(HS, -1.)
    HL = iris.analysis.maths.multiply(HL, -1.)
    ann_SEB = {'lon': real_lon, 'lat': real_lat, 'orog': orog, 'lsm': LSM , 'LWnet': LWnet[:,0,:,:],  'SWnet': SWnet[:,0,:,:],
                'LWdown': LWdown[:,0,:,:],  'SWdown': SWdown[:,0,:,:],  'HL': HL[:,0,:,:],  'HS': HS[:,0,:,:],
                'Etot': Etot[:,0,:,:], 'melt': melt_flux[:,0,:,:], 'albedo': albedo[:,0,:,:]}
    return ann_SEB

annSEB = load_ann()

def apply_Larsen_mask(var, lsm, orog):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = lsm.data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[orog.data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    # Apply mask to variable requested
    var_masked = np.ma.masked_array(var, mask=np.broadcast_to(Larsen_mask, var.shape))
    return var_masked, Larsen_mask

def ann_seas_cyc(annSEB):
    fig, ax = plt.subplots( figsize=(16,9))
    #ax = ax.flatten()
    colour_dict = {'SWdown': ('#6fb0d2', 'o'), 'SWnet': ('#b733ff', 'o'), 'SWup': ('#6fb0d2', 'o'), 'LWdown': ('#86ad63','X'), 'LWnet': ('#de2d26','X'), 'LWup': ('#86ad63','X'),'HL': ('#33aeff', '*'), 'HS': ('#ff6500', '^'), 'Etot': ('#222222', 's'), 'melt': ('#f68080', 'P')}
    leg_dict = {'SWnet': 'SW$_{net}$', 'LWnet': 'LW$_{net}$', 'HL': 'H$_{L}$', 'HS': 'H$_{S}$', 'Etot': 'E$_{tot}$', 'melt': 'E$_{melt}$'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for j in ['SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'melt']:
        j_m, Larsen_mask = apply_Larsen_mask(annSEB[j].data, annSEB['lsm'], annSEB['orog'])
        first_half = j_m[213:]
        second_half = j_m[:212]
        austral_var = np.concatenate((np.mean(first_half, axis = (1,2)), np.mean(second_half, axis = (1,2))), axis=0)
        ax.plot(austral_var, color=colour_dict[j][0], marker = colour_dict[j][1],  markersize = 10, lw=2, label = leg_dict[j])
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
    #ax.set_ylim(-60, 120)#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
    ax.set_xlim(0,366)
    ax.set_xticks(ticks=[15, 105, 225, 345])
    ax.set_xticklabels(['Aug', 'Nov', 'Mar', 'Jul'])
    #lab = ax.text(0.08, 0.85, zorder = 100, transform = ax.transAxes, s=lab_dict[0], fontsize=32, fontweight='bold', color='dimgrey')
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='major', length = 8, width = 2, color = 'dimgrey', labelsize=24, tick1On=True, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.tick_params(axis='both', which='minor', length=0, width=0, color='dimgrey', labelsize=24, tick1On=True,
                          tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    ax.yaxis.set_label_coords(-0.15, .5)
    # Legend
    lgd = ax.legend(bbox_to_anchor=(0.8, 1), markerscale=2, ncol = 2, loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    for line in lgd.get_lines():
        line.set_linewidth(4.0)
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.15, hspace = 0.05, top = 0.92, right = 0.85, left = 0.16, bottom = 0.08)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/Seasonal_cyc_SEB_climatology.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/Seasonal_cyc_SEB_climatology.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Seasonal_cyc_SEB_climatology.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Seasonal_cyc_SEB_climatology.eps')
    plt.show()

ann_seas_cyc(annSEB)

# Load ERA 5 data
lsm = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'land_binary_mask')
orog = iris.load_cube('Downloads/ERA5_orog_lsm.nc', 'geopotential')
orog = orog/9.80665 # in m

era_dict_DJF = {}
for t in ['00Z', '03Z', '06Z', '09Z', '12Z', '15Z', '18Z', '21Z']:
    era_dict_DJF[t]={}
    for v in ['surface_net_downward_shortwave_flux', 'surface_net_upward_longwave_flux', 'surface_upward_sensible_heat_flux',
              'surface_upward_latent_heat_flux', 'Snowmelt', 'Snow albedo', 'Skin temperature']:
        era_dict_DJF[t][v] = iris.load_cube('Downloads/ERA5_monthly_DJF_'+ t + '.nc', v)
        era_dict_DJF[t][v].data = np.ma.masked_where(np.broadcast_to(lsm[0].data != 1., era_dict_DJF[t][v].shape),era_dict_DJF[t][v].data)
        if v == 'surface_net_downward_shortwave_flux' or v == 'surface_net_upward_longwave_flux' or v == 'surface_upward_sensible_heat_flux' or v == 'surface_upward_latent_heat_flux':
            era_dict_DJF[t][v] = era_dict_DJF[t][v] / 86400 # convert to W m-2
    era_dict_DJF[t]['Etot'] = era_dict_DJF[t]['surface_net_downward_shortwave_flux'] - era_dict_DJF[t]['surface_net_upward_longwave_flux'] \
                       - era_dict_DJF[t]['surface_upward_sensible_heat_flux'] - era_dict_DJF[t]['surface_upward_latent_heat_flux']

inlet = copy.deepcopy(era_dict_DJF)
for t in ['00Z', '03Z', '06Z', '09Z', '12Z', '15Z', '18Z', '21Z']:
    for v in inlet[t].keys():
        # find values for points close to steep orography
        inlet[t][v] = np.ma.masked_where((np.broadcast_to(orog.data, inlet[t][v].data.shape) > 250.) & (np.broadcast_to(orog.data, inlet[t][v].data.shape) > 100.), inlet[t][v].data)
        inlet[t][v] = np.mean(inlet[t][v].data) # time series

ice_shelf = copy.deepcopy(era_dict_DJF)
for t in ['00Z', '03Z', '06Z', '09Z', '12Z', '15Z', '18Z', '21Z']:
    for v in ice_shelf[t].keys():
        #find values for ice shelf points away from steep orography
        ice_shelf[t][v] = np.ma.masked_where(np.broadcast_to(orog[0].data < 100., ice_shelf[t][v].data.shape), ice_shelf[t][v].data)
        ice_shelf[t][v] = np.mean(ice_shelf[t][v].data)

time_srs = pd.date_range(datetime(1998,1,1,0,0,0),datetime(2017,12,31,23,59,59), freq ='M')

inlet_df = pd.DataFrame()
ice_shelf_df = pd.DataFrame()

for v in inlet['00Z'].keys():
    #ice_shelf_df[v] = pd.Series(ice_shelf['00Z'][v])
    ice_shelf_df[v] = pd.Series([ice_shelf['00Z'][v], ice_shelf['03Z'][v], ice_shelf['06Z'][v], ice_shelf['09Z'][v], ice_shelf['12Z'][v], ice_shelf['15Z'][v],
         ice_shelf['18Z'][v], ice_shelf['21Z'][v]])
    inlet_df[v] = pd.Series([inlet['00Z'][v], inlet['03Z'][v],inlet['06Z'][v],inlet['09Z'][v],inlet['12Z'][v],inlet['15Z'][v],inlet['18Z'][v],inlet['21Z'][v]])

inlet_df.index = time_srs
ice_shelf_df.index = time_srs

def SEB_subplot():
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(24, 12))
    ax = ax.flatten()
    colour_dict = {'SWdown': ('#6fb0d2', 'o'), 'SWnet': ('#b733ff', 'o'), 'SWup': ('#6fb0d2', 'o'), 'LWdown': ('#86ad63','X'), 'LWnet': ('#de2d26','X'), 'LWup': ('#86ad63','X'),'HL': ('#33aeff', '*'), 'HS': ('#ff6500', '^'), 'Etot': ('#222222', 's'), 'melt': ('#f68080', 'P')}
    leg_dict = {'SWnet': 'SW$_{net}$', 'LWnet': 'LW$_{net}$', 'HL': 'H$_{L}$', 'HS': 'H$_{S}$', 'Etot': 'E$_{tot}$', 'melt': 'E$_{melt}$'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    UTC_DJF_inlet = pd.DataFrame()
    UTC_JJA_inlet = pd.DataFrame()
    UTC_DJF_shelf = pd.DataFrame()
    UTC_JJA_shelf = pd.DataFrame()
    for j in ['SWnet', 'LWnet', 'HS', 'HL', 'Etot', 'melt']:
        ice_shelf_mn = ((DJF[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + DJF[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((DJF[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + DJF[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_DJF_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_DJF_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf_mn = ((JJA[j][:, lon_dict['AWS14'], lat_dict['AWS14']].data + JJA[j][:, lon_dict['AWS15'], lat_dict['AWS15']].data) / 2.)
        inlet_mn = ((JJA[j][:, lon_dict['AWS17'], lat_dict['AWS17']].data + JJA[j][:, lon_dict['AWS18'], lat_dict['AWS18']].data) / 2.)
        UTC_JJA_inlet[j] = np.concatenate((inlet_mn[1:], inlet_mn[:2]), axis=0)
        UTC_JJA_shelf[j] = np.concatenate((ice_shelf_mn[1:], ice_shelf_mn[:2]), axis=0)
        ice_shelf = ax[0].plot(UTC_DJF_shelf[j], color=colour_dict[j][0], marker = colour_dict[j][1],  markersize = 15, lw=4, label = leg_dict[j])
        inlet = ax[1].plot(UTC_DJF_inlet[j], color=colour_dict[j][0],marker = colour_dict[j][1], markersize = 15, lw=4, label = leg_dict[j])
        ax[2].plot(UTC_JJA_inlet[j], color=colour_dict[j][0], marker = colour_dict[j][1], lw=4, markersize = 15, label = leg_dict[j])
        ax[3].plot(UTC_JJA_shelf[j], color=colour_dict[j][0], marker = colour_dict[j][1], lw=4, markersize = 15, label = leg_dict[j])
     #   AWS_srs = ax[plot].plot(AWS_dict[AWS]['datetime'],AWS_dict[AWS][j][:, lon_index18, lat_index18].data, color=col_dict[AWS], linewidth=2.5, label='MetUM output at '+ AWS)
    for i in range(4):
        #ax[i].set_xlim(DJF['Time_srs'][1], DJF['Time_srs'][-1])
        ax[i].tick_params(axis='both', which='both', labelsize=24, tick1On = False, tick2On = False)
        ax[i].set_ylim(-60, 120)#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[i].set_xlim(0,8.01)
        lab = ax[i].text(0.08, 0.85, zorder = 100, transform = ax[i].transAxes, s=lab_dict[i], fontsize=32, fontweight='bold', color='dimgrey')
        ax[i].spines['top'].set_visible(False)
        plt.setp(ax[i].spines.values(), linewidth=2, color='dimgrey', )
        ax[i].tick_params(axis='both', which='major', length = 8, width = 2, color = 'dimgrey', labelsize=24, tick1On=True, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[i].tick_params(axis='both', which='minor', length=0, width=0, color='dimgrey', labelsize=24, tick1On=True,
                          tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (w, l) in enumerate(ax[i].xaxis.get_major_ticks()) if w % 2 != 0]
        #ax[i].xaxis.get_major_ticks()[-2].set_visible(False)
    ax[0].yaxis.set_label_coords(0, 1)
    ax[1].yaxis.set_label_coords(1, 1)
    ax[0].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    ax[1].set_ylabel('Energy flux\n(W m$^{-2}$)', fontsize=24, color='dimgrey', rotation=0)
    for axs in [ax[0], ax[2]]:
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_ticks_position('right')
        #axs.tick_params(axis='y', tick2On = False, tick1On = False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        axs.yaxis.set_label(' ')
        plt.xticks(range(9), ('00:00', '03:00', '06:00','09:00', '12:00', '15:00', '18:00', '21:00', '00:00', '00:00'))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 4 != 0]
    # Legend
    lgd = ax[3].legend(bbox_to_anchor=(0.3, 1), ncol = 2, loc=2, fontsize=20)
    ax[0].set_title('Ice shelf stations', fontsize = 30, color = 'dimgrey')
    ax[0].text(-0.4, 0.55, zorder=100, transform=ax[0].transAxes, s='DJF', fontsize=32, fontweight='bold',color='dimgrey')
    ax[2].text(-0.4, 0.55, zorder=100, transform=ax[2].transAxes, s='JJA', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_title('Inlet stations', fontsize = 30, color = 'dimgrey')
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.15, hspace = 0.05, top = 0.92, right = 0.85, left = 0.16, bottom = 0.08)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/1998-2017_inlet_v_ice_shelf_seas_SEB.eps')
    elif host == 'jasmin':
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB.png')
        plt.savefig(
            '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/1998-2017_inlet_v_ice_shelf_seas_SEB.eps')
    plt.show()
