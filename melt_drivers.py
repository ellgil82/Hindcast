
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
import glob
from scipy import stats

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
    lsm_name = 'land_binary_mask'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'
    lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'

## Load data
def load_vars(year):
    # Set up filepath
    if host == 'jasmin':
        filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
        ancil_path = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/'
        lsm_name = 'land_binary_mask'
    elif host == 'bsl':
        filepath = '/data/mac/ellgil82/hindcast/output/'
        ancil_path = filepath
        lsm_name = 'LAND MASK (No halo) (LAND=TRUE)'
    try:
        melt_flux = iris.load_cube(filepath + year + '_land_snow_melt_flux.nc', 'Snow melt heating flux')  # W m-2
        melt_amnt = iris.load_cube(filepath + year + '_land_snow_melt_amnt.nc', 'Snowmelt')  # kg m-2
        melt_amnt = iris.analysis.maths.multiply(melt_amnt, 108.)
        melt_rate = iris.load_cube(filepath + year + '_Ts.nc', 'surface_temperature')
        SW_down = iris.load_cube(filepath + year + '_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
        LW_down = iris.load_cube(filepath + year + '_surface_LW_down.nc', 'IR down')
        cloud_cover = iris.load_cube(filepath + year + '_cl_frac.nc', 'Total cloud')
        IWP = iris.load_cube(filepath + year + '_total_column_ice.nc', 'atmosphere_cloud_ice_content')
        LWP = iris.load_cube(filepath + year + '_total_column_liquid.nc', 'atmosphere_cloud_liquid_water_content')
        WVP = iris.load_cube(filepath + year + '_total_column_vapour.nc')
        #melt_rate = iris.load_cube(filepath + year + '_land_snow_melt_rate.nc', 'Rate of snow melt on land')  # kg m-2 s-1
        foehn_freq = iris.load_cube(filepath + 'FI_norm.nc')
        orog = iris.load_cube(ancil_path + 'orog.nc')
        orog = orog[0, 0, :, :]
        LSM = iris.load_cube(ancil_path + 'new_mask.nc')
        LSM = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')
    var_list = [melt_rate, melt_amnt, melt_flux, SW_down, cloud_cover, IWP, LWP, WVP, LW_down]
    for i in var_list:
        real_lon, real_lat = rotate_data(i, 2, 3)
    vars_yr = {'melt_flux': melt_flux[:,0,:,:], 'melt_rate': melt_rate[:,0,:,:], 'melt_amnt': melt_amnt[:,0,:,:], 'SW_down': SW_down[:,0,:,:],  'LW_down': LW_down[:,0,:,:], 'cl_cover': cloud_cover[:,0,:,:],
               'IWP': IWP[:,0,:,:], 'LWP': LWP[:,0,:,:], 'WVP': WVP[:,0,:,:], 'foehn_freq': foehn_freq,  'orog': orog, 'lsm': LSM,'lon': real_lon, 'lat': real_lat, 'year': year}
    return vars_yr

#surf= load_vars('2012')
surf = load_vars('1998-2017')

year_list = ['1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']


lon_index14, lat_index14 = find_gridbox(-67.01, -61.03, surf['lat'], surf['lon'])
lon_index15, lat_index15 = find_gridbox(-67.34, -62.09, surf['lat'], surf['lon'])
lon_index17, lat_index17 = find_gridbox(-65.93, -61.85, surf['lat'], surf['lon'])
lon_index18, lat_index18 = find_gridbox(-66.48272, -63.37105, surf['lat'], surf['lon'])

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

## Melt index
# Define melt_percentile
melt_amnt = np.copy(surf['melt_amnt'].data)
melt_amnt = np.ma.masked_greater(melt_amnt, 500)
melt_90_pctl = np.percentile(a = melt_amnt, q = 90, axis = 0)
masked_3d_melt = np.ma.masked_less(surf['melt_amnt'].data, np.broadcast_to(melt_90_pctl, surf['melt_amnt'].shape)) # this becomes composite mask

melt_90_pctl = np.zeros((220,220))
for i in range(220):
    for j in range(220):
        melt_90_pctl[i,j] = np.percentile(melt_amnt[:,i,j].data, q = 90)

## Diagnose correlation between shortwave flux and melting across model domain
def cloud_melt(station, year_list):
    cloud_melt = pd.DataFrame(index = ['r', 'r_squared', 'p', 'std_err'])
    for year in year_list:
        SW = iris.load_cube(filepath + year + '_surface_SW_down.nc', 'surface_downwelling_shortwave_flux_in_air')
        melt = iris.load_cube(filepath + year + '_land_snow_melt_amnt.nc', 'Snowmelt')
        SW = SW[:,0,:,:]
        melt = melt[:,0,:,:]
        if station == 'ice_shelf':
            orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
            orog = orog[0, 0, :, :]
            LSM = iris.load_cube(filepath + 'new_mask.nc', lsm_name)
            lsm = LSM[0, 0, :, :]
            # Make ice shelf mask
            Larsen_mask = np.zeros((220, 220))
            lsm_subset = lsm.data[:150, 90:160]
            Larsen_mask[:150, 90:160] = lsm_subset
            Larsen_mask[orog.data > 100] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            melt_masked = np.ma.masked_array(melt.data, mask=np.broadcast_to(Larsen_mask, melt.shape)).mean(axis=(1, 2))
            SW_masked = np.ma.masked_array(SW.data, mask=np.broadcast_to(Larsen_mask, SW.shape)).mean(axis=(1, 2))
        else:
            melt_masked = melt.data[:, lat_dict[station], lon_dict[station]]
            SW_masked =SW.data[:, lat_dict[station], lon_dict[station]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(melt_masked, SW_masked)
        stats_yr = [r_value, r_value**2, p_value, std_err]
        cloud_melt[year] = pd.Series(stats_yr, index = ['r', 'r_squared', 'p', 'std_err'])
        cloud_melt.to_csv(filepath + 'cloud_v_melt_stats_model_' + station + '.csv')

#cloud_melt(station = 'AWS14', year_list= year_list)
#cloud_melt(station = 'AWS17', year_list= year_list)
#cloud_melt(station = 'AWS18', year_list = year_list)
#cloud_melt(station = 'ice_shelf', year_list= year_list)

for file in os.listdir(filepath):
    if fnmatch.fnmatch(file, 'Modelled_seasonal_foehn_frequency_%(station)s*.csv' % locals()):
        foehn_freq = pd.read_csv(str(file), na_values=-9999, header=0)
for year in year_list:
    print('Loading model data from ' + year)
    ANN = load_vars(year)
    DJF = load_vars('DJF_'+year)
    MAM = load_vars('MAM_'+year)
    JJA = load_vars('JJA_'+year)
    SON = load_vars('SON_'+year)
    total_melt_ANN = np.cumsum(ANN['melt_amnt'].data[:,lat_dict[station], lon_dict[station]], axis = 0)[-1]
    total_melt_DJF = np.cumsum(DJF['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_MAM = np.cumsum(MAM['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_JJA = np.cumsum(JJA['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]
    total_melt_SON = np.cumsum(SON['melt_amnt'].data[:, lat_dict[station], lon_dict[station]], axis=0)[-1]

stats_df = pd.DataFrame(index = ['r', 'r_squared', 'p', 'std_err'])
slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['ANN'],total_melt_ANN)
ANN_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['MAM'],total_melt_MAM)
MAM_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['DJF'],total_melt_DJF)
DJF_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['JJA'],total_melt_JJA)
JJA_stats = [r_value, r_value**2, p, std_err]

slope, intercept, r_value, p_value, std_err = stats.linregress(foehn_freq['SON'],total_melt_SON)
SON_stats = [r_value, r_value**2, p, std_err]


# Plot spatial correlations over whole domain - where is correlation between SW and melting strongest?

# at each *unmasked* gridpoint, compute correlations aver time axis

def run_corr(year_vars, xvar, yvar):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = year_vars['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[year_vars['orog'][:,:].data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    x_masked = np.ma.masked_array(year_vars[xvar].data[:58434], mask=np.broadcast_to(Larsen_mask, year_vars[xvar][:58434].shape))
    y_masked = np.ma.masked_array(year_vars[yvar].data, mask=np.broadcast_to(Larsen_mask, year_vars[yvar].shape))
    unmasked_idx = np.where(Larsen_mask == 0)
    r = np.zeros((220,220))
    p = np.zeros((220,220))
    err = np.zeros((220,220))
    for x, y in zip(unmasked_idx[0],unmasked_idx[1]):
        if x > 0. or y > 0.:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_masked[:,x, y], y_masked[:,x, y])
            r[x,y] = r_value
            p[x,y] = p_value
            err[x,y] = std_err
    r2 = r**2
    return r, r2, p, err, x_masked, y_masked, Larsen_mask

def correlation_maps(year_list, xvar, yvar):
    if len(year_list) > 1:
        fig, ax = plt.subplots(7, 3, figsize=(8, 18))
        CbAx = fig.add_axes([0.25, 0.1, 0.5, 0.02])
        ax = ax.flatten()
        comp_r = np.zeros((220, 220))
        plot = 0
        for year in year_list:
            vars_yr = load_vars(year)
            ax[plot].axis('off')
            r, r2, p, err, xmasked, y_masked, Larsen_mask = run_corr(vars_yr, xvar = xvar, yvar = yvar[:58444])
            if np.mean(r) < 0:
                squished_cmap = shiftedColorMap(cmap=matplotlib.cm.bone_r, min_val=-1., max_val=0., name='squished_cmap', var=r, start=0.25, stop=0.75)
                c = ax[plot].pcolormesh(r, cmap=matplotlib.cm.Spectral, vmin=-1., vmax=0.)
            elif np.mean(r) > 0:
                squished_cmap = shiftedColorMap(cmap = matplotlib.cm.gist_heat_r, min_val = 0, max_val = 1, name = 'squished_cmap', var = r, start = 0.25, stop = 0.75)
                c = ax[plot].pcolormesh(r, cmap = matplotlib.cm.Spectral, vmin = 0., vmax = 1.)
            ax[plot].contour(vars_yr['lsm'].data, colors='#222222', lw=2)
            ax[plot].contour(vars_yr['orog'].data, colors='#222222', levels=[100])
            comp_r = comp_r + r
            ax[plot].text(0.4, 1.1, s=year_list[plot], fontsize=24, color='dimgrey', transform=ax[plot].transAxes)
            unmasked_idx = np.where(y_masked.mask[0,:,:] == 0)
            sig = np.ma.masked_array(p, mask = y_masked[0,:,:].mask)
            sig = np.ma.masked_greater(sig, 0.01)
            ax[plot].contourf(sig, hatches = '...')
            plot = plot + 1
        mean_r_composite = comp_r / len(year_list)
        ax[-1].contour(vars_yr['lsm'].data, colors='#222222', lw=2)
        if np.mean(mean_r_composite) < 0:
            squished_cmap = shiftedColorMap(cmap=matplotlib.cm.bone_r, min_val=-1., max_val=0., name='squished_cmap',
                                            var=mean_r_composite, start=0.25, stop=0.75)
            c = ax[-1].pcolormesh(mean_r_composite, cmap=squished_cmap, vmin=-1., vmax=0.)
        elif np.mean(r) > 0:
            squished_cmap = shiftedColorMap(cmap=matplotlib.cm.gist_heat_r, min_val=0, max_val=1, name='squished_cmap',
                                            var=mean_r_composite, start=0.25, stop=0.75)
            c = ax[-1].pcolormesh(r, cmap=squished_cmap, vmin=0., vmax=1.)
        ax[-1].contour(vars_yr['orog'].data, colors='#222222', levels=[100])
        ax[-1].text(0., 1.1, s='Composite', fontsize=24, color='dimgrey', transform=ax[-1].transAxes)
        cb = plt.colorbar(c, orientation='horizontal', cax=CbAx, ticks=[0, 0.5, 1])
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        cb.outline.set_linewidth(2)
        cb.ax.xaxis.set_ticks_position('bottom')
        # cb.ax.set_xticks([0,4,8])
        cb.set_label('Correlation coefficient', fontsize=24, color='dimgrey', labelpad=30)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, hspace=0.3, wspace=0.05)
        if host == 'bsl':
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ xvar + '_v_' + yvar + '_all_years.png', transparent=True)
            plt.savefig('/users/ellgil82/figures/Hindcast/SMB/'+ xvar + '_v_' + yvar + '_all_years.eps', transparent=True)
        elif host == 'jasmin':
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/'+ xvar + '_v_' + yvar + '_all_years.png', transparent=True)
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/'+ xvar + '_v_' + yvar + '_all_years.eps', transparent=True)
    # Save composite separately
    elif len(year_list) == 1:
        r, r2, p, err, xmasked, y_masked, Larsen_mask = run_corr(surf, xvar=xvar, yvar=yvar)
    unmasked_idx = np.where(y_masked.mask[0, :, :] == 0)
    sig = np.ma.masked_array(p, mask=y_masked[0, :, :].mask)
    sig = np.ma.masked_greater(sig, 0.01)
    mean_r_composite = r
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    # Make masked areas white, instead of colour used for zero in colour map
    ax.contourf(y_masked.mask[0, :, :], cmap='Greys_r')
    # Plot coastline
    ax.contour(surf['lsm'].data, colors='#222222', lw=2)
    # Plot correlations
    c = ax.pcolormesh(np.ma.masked_where((y_masked.mask[0, :, :] == 1.), mean_r_composite), cmap=matplotlib.cm.Spectral)#, vmin=-1, vmax=1)
    # Plot 50 m orography contour on top
    ax.contour(surf['orog'].data, colors='#222222', levels=[100])
    # Overlay stippling to indicate signficance
    ax.contourf(sig, hatches='...', alpha = 0.0)
    # Set up colourbar
    cb = plt.colorbar(c, orientation='horizontal')#, ticks=[-1, 0, 1])
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                      tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label('Correlation coefficient', fontsize=24, color='dimgrey', labelpad=30)
    # Save figure
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.png', transparent=True)
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.eps', transparent=True)
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_composite.png',transparent=True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_composite.eps',transparent=True)
    plt.show()

for i in ['LW_down', 'SW_down', 'IWP', 'LWP']:
    correlation_maps(['1998-2017'], xvar = 'cl_cover', yvar = i)

for i in ['foehn_freq']:#['cl_cover', 'SW_down', 'LW_down', 'IWP', 'LWP']:
    correlation_maps(year_list = year_list, xvar = 'melt_amnt', yvar = i)

correlation_maps(['1998-2017'], xvar = 'melt_amnt', yvar = 'LW_down')

surf['foehn_freq'] = iris.load_cube(filepath + 'foehn_index_noFF.nc')
# add t+2 to end
correlation_maps(['1998-2017'], xvar = 'melt_amnt', yvar = 'foehn_freq')

melt_sum = iris.load_cube(filepath + 'melt_sum.nc')

def foehn_melt():
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = surf['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[surf['orog'][:,:].data > 100] = 0
    Larsen_mask = np.logical_not(Larsen_mask)
    x_masked = np.ma.masked_array(melt_sum.data, mask=np.broadcast_to(Larsen_mask, melt_sum.shape))
    y_masked = np.ma.masked_array(f_cond.data, mask=np.broadcast_to(Larsen_mask, f_cond.shape))
    unmasked_idx = np.where(Larsen_mask == 0)
    r = np.zeros((220,220))
    p = np.zeros((220,220))
    err = np.zeros((220,220))
    for x, y in zip(unmasked_idx[0],unmasked_idx[1]):
        if x > 0. or y > 0.:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_masked[x, y], y_masked[x, y])
            r[x,y] = r_value
            p[x,y] = p_value
            err[x,y] = std_err
    r2 = r**2
    return r, r2, p, err, x_masked, y_masked

r, r2, p, err, x_masked, y_masked = foehn_melt()
