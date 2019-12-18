# Define where the script is running
host = 'jasmin'

# Import packages
from __future__ import division
import iris
import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import iris.plot as iplt
import sys
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
from eofs.iris import Eof

# Set up filepath
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/alloutput/'
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/hindcast/output/'

MSLP = iris.load_cube(filepath + '1998-2017_MSLP.nc')

try:
    LSM = iris.load_cube(filepath+'new_mask.nc')
    orog = iris.load_cube(filepath+'orog.nc')
    orog = orog[0,0,:,:]
    lsm = LSM[0,0,:,:]
except iris.exceptions.ConstraintMismatchError:
    print('Files not found')

for i in [lsm, orog]:
    real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)

def rmv_mn(input):
    mn = input.collapsed('time', iris.analysis.MEAN)
    anom = input - mn
    return mn, anom

melt_mn = iris.load_cube(filepath + '1998-2017_land_snow_melt_amnt_daymn.nc')
cl_mn = iris.load_cube(filepath + '1998-2017_cl_frac_daymn.nc')
MSLP_daymn = iris.load_cube(filepath + '1998-2017_MSLP_daymn.nc')
#Tair_daymn = iris.load_cube(filepath + '1998-2017_Tair_daymn.nc')
FF_daymn = iris.load_cube(filepath + '1998-2017_FF_10m_daymn.nc')



for i in [cl_mn, melt_mn, MSLP_daymn]:
    real_lon, real_lat = rotate_data(i, np.ndim(i) - 2, np.ndim(i) - 1)

def calc_eofs(input, neofs):
    mn, anom = rmv_mn(input)
    solver = Eof(anom, weights = 'coslat')
    eofs = solver.eofs(neofs = neofs)
    pcs = solver.pcs(npcs = neofs)
    return solver, eofs, pcs

melt_solver, melt_eofs, melt_pcs = calc_eofs(melt_mn, neofs = 4)
cl_solver, cl_eofs, cl_pcs = calc_eofs(cl_mn, neofs = 4)

from eofs.multivariate.standard import MultivariateEof
msolver = MultivariateEof([cl_mn.data, melt_mn.data, FF_daymn.data])

# calculate 20 year daily mean climatology
#year_list = ['1998', '1999', '2000','2001', '2002', '2003', '2004', '2005', '2006', '2007',  '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
#for year in year_list:
#    iris.load_cube(filepath + year + '_MSLP_daymn.nc')


def draw_map(var):
    fig, ax = plt.subplots(figsize=(12,8), edgecolor = 'dimgrey')
    #ax = fig.add_subplot(111)
    m = Basemap(projection='stere', llcrnrlat= min(real_lat+0.2) , urcrnrlat= max(real_lat), llcrnrlon= min(real_lon), urcrnrlon= max(real_lon), lat_ts = -67, lon_0 = min(real_lon)+((max(real_lon)-min(real_lon))/2), lat_0 = min(real_lat)+((max(real_lat)-min(real_lat+0.2))/2))
    #ax.patch.set_facecolor('#566da0')
    ax.set_xlim(min(real_lon), max(real_lon))
    ax.set_ylim(min(real_lat +0.2), max(real_lat))
    xlon, ylat = np.meshgrid(real_lon,real_lat)
    lsm_masked = np.ma.masked_where(lsm.data==0, lsm.data)
    m.contourf(xlon, ylat, lsm_masked, colors='w', latlon=True, zorder  =1)
    m.contour(xlon, ylat, lsm.data, levels = [0], colors='#222222', lw = 2, latlon=True, zorder=2)
    m.contour(xlon, ylat, orog.data, levels = [10], colors = '#222222', linewidth = 1.5, latlon= True, zorder= 3)
    if var == MSLP:
        eof1 = m.contour(xlon, xlat, eofs[0, 0].data, colors='#222222', linewidth=2.5, latlon=True, zorder=6)
        m.clabel(eof1, inline=True, fontsize=24)
    else:
        m.contour(xlon, ylat, eofs[0,0].data, latlon = True)
    # Get the colormap colors, multiply them by 0.75, and create new colormap
    my_cmap = plt.cm.copper_r(np.arange(plt.cm.copper_r.N))
    my_cmap[:, 0:3] *= 0.75
    my_cmap = ListedColormap(my_cmap)
    m.contourf(xlon, ylat, np.ma.masked_where(orog.data < 10, orog.data), cmap = my_cmap, latlon = True, vmin = 0, vmax = 5000)
    merid = m.drawmeridians(meridians=np.arange(np.around(min(real_lon)),np.around(max(real_lon)),6), labels = [False,False, True,False,True,True], fontsize =30 , color = '#222222')
    parallels=np.arange(np.around(min(real_lat)),np.around(max(real_lat)),2)
    m.drawmapscale(max(real_lon)-2,min(real_lat)+0.6,-70,-67.54,200,'fancy','km', fontsize = 24, fillcolor1 = 'w', fillcolor2 = '#222222', fontcolor = '#222222')
    par = m.drawparallels(parallels, labels=[True, False,True,False],fontsize =30, color = '#222222' )
    m.drawmapboundary(color='dimgrey', linewidth=2, fill_color='#566da0', ax = ax)
    #m.scatter(-62.09, -67.34, s= 300, latlon= True, marker = 'X', color = '#4C9900', zorder = 20)
    #m.scatter(-61.85, -65.93, s= 300, latlon= True, marker = 'X', color = '#4C9900', zorder = 20)
    #m.scatter(-66.48, -63.37, s= 300, latlon= True, marker = 'X', color = '#4C9900', zorder = 20)
    #m.scatter(-61.03, -67.01, s= 300, latlon= True, marker = 'X', color = '#4C9900', zorder = 20)
    #ax.annotate(xy=(0.5, 0.54), s='AWS 14', fontsize='30', color='#222222', xycoords='axes fraction')
    #ax.annotate(xy=(0.5, 0.4), s='AWS 15', fontsize='30', color='#222222', xycoords='axes fraction')
    #ax.annotate(xy=(0.5, 0.8), s='AWS 17', fontsize='30', color='#222222', xycoords='axes fraction')
    #ax.annotate(xy=(0.5, 0.3), s='AWS 18', fontsize='30', color='#222222', xycoords='axes fraction')
    # Change colours of labels
    def set_colour(x, colour):
        for m in x:
            for t in x[m][1]:
                t.set_color(colour)
    set_colour(par, 'dimgrey')
    set_colour(merid, 'dimgrey')
    # Draw 'cloud' box
    if draw_track == 'yes' or 'Y' or 'y':
        #draw_screen_poly(latbox, lonbox, m)
        cm = plt.get_cmap('magma')
        cNorm = matplotlib.colors.Normalize(vmin=0,vmax = 5000) # max(alt)+500)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        m.scatter(lon.data,lat.data,  s= 100, color = scalarMap.to_rgba(alt.data),latlon=True, zorder =4)
        scalarMap.set_array(alt)
        divider = make_axes_locatable(ax)
        cax=divider.append_axes('right', size='5%', pad=0.7)
        cb = fig.colorbar(scalarMap, cax=cax, )
        [l.set_visible(False) for (i, l) in enumerate(cb.ax.xaxis.get_ticklabels()) if i % 2 != 0]
        cb.ax.tick_params(labelsize=24)
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=30, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                            tick2On=False)
        cb.outline.set_linewidth(2)
        cb.ax.text(-2.5, 1.05, 'Altitude (m)', rotation = 0, fontsize=28, color='dimgrey')
        cb.ax.set_yticks([0, 5000])
        [l.set_visible(False) for (i, l) in enumerate(cb.ax.yaxis.get_ticklabels()) if i % 2 != 0]
        plt.subplots_adjust(left = 0.1, bottom = 0.05)
        save_direc = '/users/ellgil82/figures/Cloud data/f' + str(case) + '/'
        config = 'flight_track_with_cloud_box'
        plt.savefig(save_direc + case + '_' + config + '.eps')
        plt.savefig(save_direc + case + '_' + config + '.png')
    elif draw_track == 'no' or 'N' or 'n':
        plt.subplots_adjust(left=0.15, bottom=0.05)
        plt.savefig(filepath + 'EOF_1.eps')
        plt.savefig(filepath + 'EOF_1.eps')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    plt.show()

draw_map()

fig, ax = plt.subplots(2,2)
ax = ax.flatten()
for i in range(4):
    e = ax[i].pcolormesh(melt_eofs[i, 0,:,:].data)#, vmin = -0.01, vmax = 0.01)

plt.colorbar(e, orientation = 'horizontal')
plt.show()