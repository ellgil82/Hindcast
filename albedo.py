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

from rotate_data import rotate_data

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/ceda_archive/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

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

seas = ''

#SWnet = iris.load_cube(filepath + seas + '1998-2017_surface_SW_net_daymn.nc', 'Net short wave radiation flux')
#SWdown = iris.load_cube(filepath + seas + '1998-2017_surface_SW_down_daymn.nc', 'surface_downwelling_shortwave_flux_in_air')
#melt_flux = iris.load_cube(filepath + seas + '1998-2017_land_snow_melt_flux_daymn.nc', 'Snow melt heating flux')

SWdown = iris.load_cube(filepath + seas + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_surface_downwelling_shortwave_flux.nc')
SWnet = iris.load_cube(filepath + seas + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_surface_net_downward_shortwave_flux.nc')
melt_flux = iris.load_cube(filepath + seas + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_surface_snow_melt_flux.nc')
Ts = iris.load_cube(filepath + seas + 'MetUM_v11p1_Antarctic_Peninsula_4km_19980101-20171231_surface_temperature.nc')


if host == 'jasmin':
    try:
        LSM = iris.load_cube(filepath + 'new_mask.nc', 'land_binary_mask')
        orog = iris.load_cube(filepath + 'orog.nc', 'surface_altitude')
        orog = orog[0, 0, :, :]
        lsm = LSM[0, 0, :, :]
    except iris.exceptions.ConstraintMismatchError:
        print('Files not found')

# Rotate data onto standard lat/lon grid and update times
SWnet = SWnet[:,0,:,:]
SWdown = SWdown[:,0,:,:]
melt_flux = melt_flux[:,0,:,:]
for i in [SWnet, SWdown, melt_flux]:
    real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)

for i in [SWnet, SWdown, melt_flux]:
    i.attributes = {'north_pole': [296.  ,  22.99], 'name': 'solar', 'title': 'Net short wave radiation flux', 'CDO': 'Climate Data Operators version 1.9.5 (http://mpimet.mpg.de/cdo)', 'CDI': 'Climate Data Interface version 1.9.5 (http://mpimet.mpg.de/cdi)', 'Conventions': 'CF-1.6', 'source': 'Unified Model Output (Vn11.1):', 'time': '12:00', 'date': '31/12/97'}

MAM16 = {}
for n, i in zip(['SWnet', 'SWdown','melt', 'Ts'],[SWnet, SWdown, melt_flux, Ts]):
    MAM16[n] = i[53072:53807]

MAM16['SWnet'] = MAM16['SWnet'][:,0]
MAM16['albedo'] = (MAM16['SWnet'].data - MAM16['SWdown'].data) * -1. / MAM16['SWdown'].data
MAM16['albedo'][MAM16['albedo'] == 0] = np.nan
MAM16['albedo'] = iris.cube.Cube(data=MAM16['albedo'], long_name = 'Surface albedo', var_name = 'albedo',)

for n in range(3):
    MAM16['albedo'].add_dim_coord(SWnet[53072:53807].dim_coords[n],n)

MAM16['albedo'].add_aux_coord(SWnet.aux_coords[0], 0)
albedo_m, mask = apply_Larsen_mask(MAM16['albedo'].data, LSM, orog)
melt_m, mask = apply_Larsen_mask(melt_flux.data, LSM, orog)

mn_a = np.nanmean(albedo_m, axis = (1,2))
mn_m = np.nanmean(melt_m, axis = (1,2))

from itertools import chain

fig, ax = plt.subplots(figsize = (8,6))
slope, intercept, r2, p, sterr = scipy.stats.linregress(mn_a, mn_m)
if p <= 0.01:
    ax.text(0.9, 0.85, horizontalalignment='right', verticalalignment='top',
                  s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax.transAxes,
                  size=24, color='dimgrey')
else:
    ax.text(0.9, 0.85, horizontalalignment='right', verticalalignment='top',
                  s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax.transAxes, size=24,
                  color='dimgrey')

ax.scatter(mn_a, mn_m, color='#f68080', s=50)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
#ax.set(adjustable='box-forced', aspect='equal')
ax.tick_params(axis='both', which='both', labelsize=24, width=2, length=5, color='dimgrey', labelcolor='dimgrey', pad=10)
ax.yaxis.set_label_coords(-0.4, 0.5)
ax.set_xlim(0.75, 1.0)
ax.set_ylim(0, 50)
#ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c = 'k', alpha = 0.8)
ax.set_xticks(0.8, 1.)
ax.set_yticks(0,60)
ax.set_xlabel('Modelled surface albedo', size = 20, color = 'dimgrey', rotation = 0, labelpad = 10)
ax.set_ylabel('Modelled\nmelt flux', size = 20, color = 'dimgrey', rotation =0, labelpad= 80)
plt.subplots_adjust(left = 0.35, bottom=0.25)
plt.savefig(filepath + '../../figures/Albedo_melt_scatter.png')
plt.savefig(filepath + '../../figures/Albedo_melt_scatter.eps')
plt.show()


