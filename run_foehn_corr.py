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

surf = {}
surf['lsm'] = iris.load_cube(filepath + 'new_mask.nc')
surf['lsm'] = surf['lsm'][0,0,:,:]
surf['orog'] = iris.load_cube(filepath + 'orog.nc')
surf['orog'] = surf['orog'][0,0,:,:]
surf['foehn_idx'] = iris.load_cube(filepath+ 'FI_noFF_calc_grad.nc')
surf['foehn_idx'].data[surf['foehn_idx'].data == np.nan] = 0.
surf['melt_amnt'] = iris.load_cube(filepath + '1998-2017_land_snow_melt_amnt.nc')

def foehn_melt(melt_var, foehn_var):
    # Make ice shelf mask
    Larsen_mask = np.zeros((220, 220))
    lsm_subset = surf['lsm'].data[:150, 90:160]
    Larsen_mask[:150, 90:160] = lsm_subset
    Larsen_mask[surf['orog'][:,:].data > 1500] = 0
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

r, r2, p, err, x_masked, y_masked = foehn_melt(surf['melt_amnt'][4:58440, 0], surf['foehn_idx'])
np.savetxt(filepath + 'foehn_index_melt_correlation_r.csv', r, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_r2.csv', r2, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_p.csv', p, delimiter = ',')
np.savetxt(filepath + 'foehn_index_melt_correlation_err.csv', err, delimiter = ',')



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
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/'+ xvar + '_v_' + yvar + '_all_years.png', transparent=True)
            plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/'+ xvar + '_v_' + yvar + '_all_years.eps', transparent=True)
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
    c = ax.pcolormesh(np.ma.masked_where((y_masked.mask[0, :, :] == 1.), mean_r_composite), cmap=matplotlib.cm.Spectral_r)#, vmin=-1, vmax=1)
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
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/' + xvar + '_v_' + yvar + '_composite.png',transparent=True)
        plt.savefig('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/figures/Drivers/' + xvar + '_v_' + yvar + '_composite.eps',transparent=True)
    #plt.show()

correlation_maps(['1998-2017'],xvar = foehn_melt(surf['melt_amnt'][4:58440],surf['foehn_idx']))


