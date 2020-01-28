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
        foehn_idx = iris.load_cube(filepath + 'FI_noFF_calc_grad.nc')
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
               'IWP': IWP[:,0,:,:], 'LWP': LWP[:,0,:,:], 'WVP': WVP[:,0,:,:], 'foehn_idx': foehn_idx,  'orog': orog, 'lsm': LSM,'lon': real_lon, 'lat': real_lat, 'year': year}
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

def run_corr(xvar, yvar):
    time_len = xvar.shape[0]
    x_masked = np.zeros((time_len, 220,220))
    y_masked = np.zeros((time_len, 220,220))
    x_masked[:, 40:135, 90:155] = xvar[:, 40:135, 90:155]
    y_masked[:, 40:135, 90:155] = yvar[:, 40:135, 90:155]
    r = np.zeros((220,220))
    p = np.zeros((220,220))
    err = np.zeros((220,220))
    for x in range(40,135):
        for y in range(90,155):
            if x > 0. or y > 0.:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_masked[:,x, y], y_masked[:,x, y])
                r[x,y] = r_value
                p[x,y] = p_value
                err[x,y] = std_err
    r2 = r**2
    Larsen_mask = np.zeros((220,220))
    Larsen_mask[40:135, 90:155] = 1.
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
            r, r2, p, err, xmasked, y_masked, Larsen_mask = run_corr(xvar = surf[xvar], yvar = surf[yvar])
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
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/'+ xvar + '_v_' + yvar + '_all_years.png', transparent=True)
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/'+ xvar + '_v_' + yvar + '_all_years.eps', transparent=True)
    # Save composite separately
    elif len(year_list) == 1:
        #print('')
        r, r2, p, err, xmasked, y_masked, Larsen_mask = run_corr(xvar=surf[xvar].data, yvar=surf[yvar].data)
    #unmasked_idx = np.where(Larsen_mask == 1)
    sig = np.ma.masked_where((Larsen_mask == 0.), p)
    sig = np.ma.masked_greater(sig, 0.1)
    mean_r_composite = r
    fig, ax = plt.subplots(figsize=(9,12))
    ax.axis('off')
    # Make masked areas white, instead of colour used for zero in colour map
    #ax.contourf(Larsen_mask, cmap='Greys_r')
    # Plot coastline
    ax.contour(surf['lsm'].data, colors='#222222', lw=2)
    # Plot correlations
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.Spectral_r, min_val=-0.02, max_val=0.1, name='spectral_zero', var=mean_r_composite[40:135, 90:155], start=0.15, stop=0.85)
    c = ax.pcolormesh(np.ma.masked_where((Larsen_mask == 0.), mean_r_composite), cmap = bwr_zero, vmin=0, vmax=0.1)# cmap=matplotlib.cm.Spectral_r,
    # Plot 50 m orography contour on top
    ax.contour(surf['orog'].data, colors='#222222', levels=[100])
    # Overlay stippling to indicate signficance
    ax.contourf(sig, color = 'dimgrey', hatches='....', alpha = 0.4)
    # Set up colourbar
    cbax = fig.add_axes([0.3, 0.2, 0.4, 0.03])
    cb = plt.colorbar(c, cax = cbax, orientation='horizontal', ticks=[-0.02, 0, 0.05, 0.1], extend = 'both')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=28, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.set_label('Correlation coefficient', fontsize=32, color='dimgrey', labelpad=30)
    plt.subplots_adjust(bottom = 0.3, top = 0.95)
    if yvar == 'SON_FI':
        ax.text(0., 0.95, zorder=6, transform=ax.transAxes, s='b', fontsize=36, fontweight='bold', color='dimgrey')
    elif yvar == 'MAM_FI':
        ax.text(0., 0.95, zorder=6, transform=ax.transAxes, s='a', fontsize=36, fontweight='bold', color='dimgrey')
    # Save figure
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.png')
        plt.savefig('/users/ellgil82/figures/Hindcast/SMB/' + xvar + '_v_' + yvar + '_composite.eps')
    elif host == 'jasmin':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_composite.png')
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/' + xvar + '_v_' + yvar + '_composite.eps')
    plt.show()


correlation_maps(year_list = ['1998-2017'], xvar = 'SON_melt', yvar = 'SON_FI')
correlation_maps(year_list = ['1998-2017'], xvar = 'MAM_melt', yvar = 'MAM_FI')


for i in ['LW_down', 'SW_down', 'IWP', 'LWP']:
    correlation_maps(['1998-2017'], xvar = 'cl_cover', yvar = i)

for i in ['foehn_idx']:#['cl_cover', 'SW_down', 'LW_down', 'IWP', 'LWP']:
    correlation_maps(year_list = year_list, xvar = 'melt_amnt', yvar = i)

correlation_maps(['1998-2017'], xvar = 'melt_amnt', yvar = 'LW_down')

surf['foehn_freq'] = iris.load_cube(filepath + 'foehn_index_noFF.nc')
# add t+2 to end
correlation_maps(['1998-2017'], xvar = 'melt_amnt', yvar = 'foehn_freq')

melt_sum = iris.load_cube(filepath + 'melt_sum.nc')

surf = {}
surf['lsm'] = iris.load_cube(filepath + 'new_mask.nc')
surf['lsm'] = surf['lsm'][0,0,:,:]
surf['orog'] = iris.load_cube(filepath + 'orog.nc')
surf['orog'] = surf['orog'][0,0,:,:]
surf['foehn_idx'] = iris.load_cube(filepath+ 'FI_for_corr_daymn.nc')
surf['melt_amnt'] = iris.load_cube(filepath + 'melt_amnt_for_corr_daymn.nc')
FI_melt_corr_mask = np.zeros((220,220))
FI_melt_corr_mask[40:135, 90:155] = surf['foehn_idx'][0,40:135, 90:155].data
mn_foehn = np.nanmean(surf['foehn_idx'].data, axis = 0 )
DJF_FI = iris.load_cube(filepath + 'DJF_FI_for_corr_daymn.nc')
surf['MAM_FI'] = iris.load_cube(filepath + 'MAM_FI_for_corr_daymn.nc')
JJA_FI = iris.load_cube(filepath + 'JJA_FI_for_corr_daymn.nc')
surf['SON_FI'] = iris.load_cube(filepath + 'SON_FI_for_corr_daymn.nc')
surf['SON_melt'] = iris.load_cube(filepath + 'SON_melt_amnt_for_corr_daymn.nc')
surf['MAM_melt'] = iris.load_cube(filepath + 'MAM_melt_amnt_for_corr_daymn.nc')
correlation_maps(year_list = ['1998-2017'], xvar = 'SON_melt', yvar = 'SON_FI')
correlation_maps(year_list = ['1998-2017'], xvar = 'MAM_melt', yvar = 'MAM_FI')

r, r2, p, err, x_masked, y_masked = run_corr(surf['melt_amnt'].data, surf['foehn_idx'].data)


def foehn_melt(melt_var, foehn_var):
    # Make ice shelf mask
    time_len = melt_var.shape[0]-1
    Larsen_mask = np.zeros((time_len, 220, 220))
    Larsen_mask[time_len, 40:135, 90:155] = surf['lsm'].data[40:135, 90:155]
    Larsen_mask = np.logical_not(Larsen_mask)
    x_masked = np.ma.masked_array(melt_var.data, mask=np.broadcast_to(Larsen_mask, melt_var.shape))
    y_masked = np.ma.masked_array(foehn_var.data, mask=np.broadcast_to(Larsen_mask, foehn_var.shape))
    unmasked_idx = np.where(x_masked.mask == 0)
    r = np.zeros((220,220))
    p = np.zeros((220,220))
    err = np.zeros((220,220))
    for x, y in zip(unmasked_idx[1],unmasked_idx[2]):
        if x > 0. or y > 0.:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_masked[:, x, y], y_masked[:,  x, y])
            r[x,y] = r_value
            p[x,y] = p_value
            err[x,y] = std_err
    r2 = r**2
    return r, r2, p, err, x_masked, y_masked

r, r2, p, err, x_masked, y_masked = foehn_melt(surf['melt_amnt'][4:58440], surf['foehn_idx'])
np.savetxt(filepath + 'foehn_index_melt_correlation_r.csv', r, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_r2.csv', r2, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_p.csv', p, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_err.csv', err, delimiter = ',')




real_lon, real_lat = rotate_data(surf['lsm'], 0,1)
real_lon, real_lat = rotate_data(surf['orog'], 0,1)





r, r2, p, err, x_masked, y_masked = foehn_melt(SON_melt.data, SON_FI.data)

correlation_maps(['1998-2017'], )



def plot_foehn_index(subplot):
    Larsen_box = np.zeros((220, 220))
    Larsen_box[40:135, 90:155] = 1.
    if subplot == False or subplot == 'no':
        fig = plt.figure(frameon=False, figsize=(8, 8))  # !!change figure dimensions when you have a larger model domain
        fig.patch.set_visible(False)
        ax = fig.add_subplot(111)#, projection=ccrs.PlateCarree())
        plt.axis = 'off'
        #cf_var = np.nanmean(surf['foehn_idx'].data, axis = 0)
        ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s='e', fontsize=36, fontweight='bold',color='dimgrey')
        ax.text(0.4, 1.1, transform=ax.transAxes, s='ANN', fontsize=30, fontweight='bold', color='dimgrey')
        cf_var = mn_foehn
        plt.setp(ax.spines.values(), linewidth=0, color='dimgrey')
        PlotLonMin = np.min(real_lon)
        PlotLonMax = np.max(real_lon)
        PlotLatMin = np.min(real_lat)
        PlotLatMax = np.max(real_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
        plt.xticks(XTicks, XTickLabels)
        ax.set_xlim(PlotLonMin, PlotLonMax)
        ax.tick_params(which='both', pad=10, labelsize = 34, color = 'dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 3)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.yticks(YTicks, YTickLabels)
        ax.set_ylim(PlotLatMin, PlotLatMax)
        ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=30,  size=0, tick1On=False, tick2On=False)
        bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-0.25, max_val=0.25, name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85) #-6, 3
        xlon, ylat = np.meshgrid(real_lon, real_lat)
        c = ax.pcolormesh(xlon, ylat, np.ma.masked_where((Larsen_box == 0.), cf_var), cmap = 'Spectral_r', vmin = -.25, vmax = .25)
        #ax.contour(xlon, ylat, shaded, levels= [1.], colors='dimgrey', linewidths = 4, latlon=True)
        #ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
        coast = ax.contour(xlon, ylat, surf['lsm'].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
        topog = ax.contour(xlon, ylat, surf['orog'].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
    elif subplot == True or subplot == 'yes':
        fig, axs = plt.subplots(2,2,frameon=False, figsize=(15, 17))  # !!change figure dimensions when you have a larger model domain
        fig.patch.set_visible(False)
        axs = axs.flatten()
        lab_dict = {0:('a','DJF'), 1: ( 'b', 'MAM'), 2: ('c', 'JJA'), 3:('d', 'SON')}
        plot = 0
        vars = [np.nanmean(DJF_FI.data, axis = 0),np.nanmean(MAM_FI.data, axis = 0),np.nanmean(JJA_FI.data, axis = 0), np.nanmean(SON_FI.data, axis=0)]
        for ax in axs:
            plt.sca(ax)
            cf_var = vars[plot]
            plt.axis = 'off'
            plt.setp(ax.spines.values(), linewidth=0, color='dimgrey')
            PlotLonMin = np.min(real_lon)
            PlotLonMax = np.max(real_lon)
            PlotLatMin = np.min(real_lat)
            PlotLatMax = np.max(real_lat)
            XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
            XTickLabels = [None] * len(XTicks)
            for i, XTick in enumerate(XTicks):
                if XTick < 0:
                    XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
                else:
                    XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')#
            plt.xticks(XTicks, XTickLabels)
            ax.set_xlim(PlotLonMin, PlotLonMax)
            ax.tick_params(which='both', pad=10, labelsize = 34, color = 'dimgrey')
            YTicks = np.linspace(PlotLatMin, PlotLatMax, 3)
            YTickLabels = [None] * len(YTicks)
            for i, YTick in enumerate(YTicks):
                if YTick < 0:
                    YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
                else:
                    YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
            plt.yticks(YTicks, YTickLabels)
            ax.set_ylim(PlotLatMin, PlotLatMax)
            ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=30, size=0, tick1On=False, tick2On=False)
            bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-.25, max_val=.25, name='bwr_zero', var=cf_var.data, start=0.15, stop=0.85) #-6, 3
            xlon, ylat = np.meshgrid(real_lon, real_lat)
            c = ax.pcolormesh(xlon, ylat, np.ma.masked_where((Larsen_box == 0.),cf_var), cmap = 'Spectral_r', vmin = -.25, vmax = .25)
            #ax.contour(xlon, ylat, shaded, levels= [1.], colors='dimgrey', linewidths = 4, latlon=True)
            ax.text(0., 1.1, zorder=6, transform=ax.transAxes, s=lab_dict[plot][0], fontsize=32, fontweight='bold', color='dimgrey')
            ax.text(0.4,1.1, transform=ax.transAxes, s=lab_dict[plot][1], fontsize=28, fontweight='bold', color='dimgrey')
            coast = ax.contour(xlon, ylat, surf['lsm'].data, levels=[0], colors='#222222', lw=2, latlon=True, zorder=2)
            topog = ax.contour(xlon, ylat, surf['orog'].data, levels=[50], colors='#222222', linewidth=1.5, latlon=True, zorder=3)
            plot = plot+1
    CBarXTicks = [-.25, 0, .25]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.35, 0.18, 0.4, 0.02])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks, extend = 'both')  #
    CBar.set_label('Mean foehn index \n(dimensionless)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.2, bottom = 0.3, wspace = 0.35, hspace = 0.35, right = 0.85)
    if subplot == True or subplot == 'yes':
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_foehn_index_seas.png', transparent=True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_foehn_index_seas.eps', transparent=True)
    else:
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_foehn_index.png', transparent = True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Mean_foehn_index.eps', transparent = True)
    plt.show()

plot_foehn_index(subplot = False)

def chop_foehn_index():
    zero_idx = -1
    djf_idx = 717
    mam_idx = 1445
    jja_idx = 2181
    son_idx = 2917
    DJF_FI = surf['foehn_idx'].data[zero_idx + 1:djf_idx]
    MAM_FI = surf['foehn_idx'].data[djf_idx + 1:mam_idx]
    JJA_FI = surf['foehn_idx'].data[mam_idx + 1:jja_idx]
    SON_FI = surf['foehn_idx'].data[jja_idx + 1:son_idx]
    for yr in range(20)[1:]:
        DJF_FI = np.concatenate((DJF_FI, surf['foehn_idx'].data[zero_idx + 1:djf_idx]), axis = 0)
        MAM_FI = np.concatenate((MAM_FI,surf['foehn_idx'].data[djf_idx + 1:mam_idx]), axis = 0)
        JJA_FI = np.concatenate((JJA_FI,surf['foehn_idx'].data[mam_idx + 1:jja_idx]), axis = 0)
        SON_FI = np.concatenate((SON_FI,surf['foehn_idx'].data[jja_idx + 1:son_idx]), axis = 0)
        djf_idx = djf_idx + 2920
        mam_idx = mam_idx + 2920
        jja_idx = jja_idx + 2920
        son_idx = son_idx + 2920
        zero_idx = zero_idx + 2920
    return DJF_FI, MAM_FI, JJA_FI, SON_FI

#DJF_FI, MAM_FI, JJA_FI, SON_FI = chop_foehn_index()

DJF_FI_cube = iris.cube.Cube(DJF_FI)
MAM_FI_cube = iris.cube.Cube(MAM_FI)
JJA_FI_cube = iris.cube.Cube(JJA_FI)
SON_FI_cube = iris.cube.Cube(SON_FI)

iris.save(DJF_FI_cube, filepath + 'DJF_foehn_index.nc')
iris.save(MAM_FI_cube, filepath + 'MAM_foehn_index.nc')
iris.save(JJA_FI_cube, filepath + 'JJA_foehn_index.nc')
iris.save(SON_FI_cube, filepath + 'SON_foehn_index.nc')


