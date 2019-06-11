# load the data into a cubelist, without attempting to merge based on scalar dimensions
MSLP_2008_free_raw = iris.load_raw('/group_workspaces/jasmin4/bas_climate/users/ellgil82/hindcast/*T0000Z_Peninsula_4km_free_run_pa*.pp','air_pressure_at_sea_level')
# copy the cubes to a new cubelist, with each cube having a 1-element time dimension plus auxiliary coordinates for forecast_reference_time and forecast_period
cl = iris.cube.CubeList()
for cube in P_Jan:
    new_cube = iris.util.new_axis(cube, 'time')
    for coord_name in ['forecast_period', 'forecast_reference_time']:
        coord = new_cube.coord(coord_name)
        new_cube.remove_coord(coord_name)
        new_cube.add_aux_coord(coord, new_cube.coord_dims('time')[0])
    if new_cube.coord('forecast_period').points[0] != 0:
        cl.append(new_cube)

# now concatenate the new cubelist into a single cube
P_Jan = cl.concatenate_cube()
iris.save( (MSLP_2008_free), '/group_workspaces/jasmin4/bas_climate/users/ellgil82/MSLP_2008_free.nc', netcdf_format='NETCDF3_CLASSIC')


# load the data into a cubelist, without attempting to merge based on scalar dimensions
Ts_raw= iris.load_raw('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/*pa*.pp', 'surface_temperature')
# copy the cubes to a new cubelist, with each cube having a 1-element time dimension plus auxiliary coordinates for forecast_reference_time and forecast_period
cl = iris.cube.CubeList()
for cube in Ts_raw:
    new_cube = iris.util.new_axis(cube, 'time')
    for coord_name in ['forecast_period', 'forecast_reference_time']:
        coord = new_cube.coord(coord_name)
        new_cube.remove_coord(coord_name)
        new_cube.add_aux_coord(coord, new_cube.coord_dims('time')[0])
    if new_cube.coord('forecast_period').points[0] != 0:
        cl.append(new_cube)

# now concatenate the new cubelist into a single cube
Ts_OFCAP = cl.concatenate_cube()
iris.save( (MSLP_OFCAP), '/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/MSLP_OFCAP.nc', netcdf_format='NETCDF3_CLASSIC')


# -------
## N.B. for monsoon type module load scitools/experimental-current before starting python

''' Script to process multi-year model runs into single netCDF files containing time series of one variable each. 

Author: Ella Gilbert, 2019. Adapted from code supplied by James Pope. '''

# Import packages
import iris
import os
import fnmatch

# Define dictionary of standard strings that iris will look for when loading variables
long_name_dict = {'sfc_P': 'surface_air_pressure',
'Ts': 'surface_temperature',
'MSLP': 'air_pressure_at_sea_level',
'u_wind': 'x_wind',
'v_wind': 'y_wind',
'T_air': 'air_temperature', 
'q': 'specific_humidity',
'theta': 'air_potential_temperature', 
'QCF': 'mass_fraction_of_cloud_ice_in_air',
'QCL': 'mass_fraction_of_cloud_liquid_water_in_air',
}

# Define function to load individual variables, amend the time dimension, and return a single cube.
def load_var(var):
	pa = []
	for file in os.listdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/'):
	    if fnmatch.fnmatch(file, '*pa000.pp'):
	        pa.append(file)
	os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/')
	raw = iris.load_raw(pa, long_name_dict[var])
	# copy the cubes to a new cubelist, with each cube having a 1-element time dimension plus auxiliary coordinates for forecast_reference_time and forecast_period
	cl = iris.cube.CubeList()
	for cube in raw:
	    new_cube = iris.util.new_axis(cube, 'time')
	    for coord_name in ['forecast_period', 'forecast_reference_time']:
	        coord = new_cube.coord(coord_name)
	        new_cube.remove_coord(coord_name)
	        new_cube.add_aux_coord(coord, new_cube.coord_dims('time')[0])
	    if new_cube.coord('forecast_period').points[0] != 0:
	        cl.append(new_cube)
	combo_cube = cl.concatenate_cube()
	return combo_cube

# Load and save variables into files
#sfc_P = load_var('sfc_P')
Ts = load_var('Ts')
iris.save((sfc_P), '/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP_sfc_P.nc', netcdf_format = 'NETCDF3_CLASSIC')

## CDO commands to turn these raw 3-hourly files into a) daily means, and b) monthly means:
# a) cdo daymean infile.nc outile.nc
# b) cdo monmean infile.nc outile.nc