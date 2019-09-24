
files_to_load= []
for r, d, f in os.walk(os.getcwd()):
	for file in f:
		if file.endswith(".pp"):
			files_to_load.append(os.path.join(r, file))

var_files = []
for i in files_to_load:
	if fnmatch.fnmatch(i, '*2009*'):
		var_files.append(i)

Cube_list = iris.load(var_files)

for i in range(len(Cube_list)):
	Cube_list[i] = Cube_list[i][:366, :,:,:]
	save_string = Cube_list[i].attributes['history'][62:-60]
	print save_string

	iris.save(Cube_list[i], save_string)

os.chdir('/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/rerun/')
pa_cubes = iris.load('*_pa000.pp')
pb_cubes = iris.load('*_pb000.pp')
pc_cubes = iris.load('*_pc000.pp')
pd_cubes = iris.load('*_pd000.pp')
#pe_cubes = iris.load('*_pe000.pp')


iris.save(pa_cubes[0], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_P.nc')
iris.save(pa_cubes[1], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_LW_down.nc')
iris.save(pa_cubes[2], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_SW_down.nc')
iris.save(pa_cubes[3], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_LW_net.nc')
iris.save(pa_cubes[4], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_latent_heat.nc')
iris.save(pa_cubes[5], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_sensible_heat.nc')
iris.save(pa_cubes[6], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_incoming_SW.nc')
iris.save(pa_cubes[7], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_outgoing_LW.nc')
iris.save(pa_cubes[8], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_outgoing_SW.nc')


iris.save(pc_cubes[0], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_SW_net.nc')
iris.save(pc_cubes[1], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_LW_down.nc')
iris.save(pc_cubes[2], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_SW_down.nc')
iris.save(pc_cubes[3], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_LW_net.nc')
iris.save(pc_cubes[4], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_latent_heat.nc')
iris.save(pc_cubes[5], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_sensible_heat.nc')
iris.save(pc_cubes[6], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_incoming_SW.nc')
iris.save(pc_cubes[7], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_outgoing_LW.nc')
iris.save(pc_cubes[8], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_TOA_outgoing_SW.nc')

iris.save(pd_cubes[0], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_evaporation_rate.nc')
iris.save(pd_cubes[1], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_land_snow_melt_amount.nc')
iris.save(pd_cubes[2], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_land_snow_melt_flux.nc')
iris.save(pd_cubes[3], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_land_snow_melt_rate.nc')
iris.save(pd_cubes[10], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_subsurface_runoff_rate.nc')
iris.save(pd_cubes[11], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_surface_runoff_rate.nc')
iris.save(pd_cubes[12], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_potential_evaporation_rate.nc')
iris.save(pd_cubes[13], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_sublimation_rate.nc')


iris.save(pe_cubes[0], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_heavyside_function.nc')
iris.save(pe_cubes[1], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_theta_full_profile.nc')
iris.save(pe_cubes[2], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_Tair_P_levs.nc')
iris.save(pe_cubes[3], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_q_full_profile.nc')
iris.save(pe_cubes[4], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_u_wind_P_levs.nc')
iris.save(pe_cubes[5], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_u_wind_full_profile.nc')
iris.save(pe_cubes[6], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_v_wind_P_levs.nc')
iris.save(pe_cubes[7], '/gws/nopw/j04/bas_climate/users/ellgil82/hindcast/output/1999/1999_v_wind_full_profile.nc')

iris.save(pb_cubes[0], '1999_cl_frac.nc')
iris.save(pb_cubes[1], '1999_water_vapour_path.nc')
iris.save(pb_cubes[2], '1999_ice_water_path.nc')
iris.save(pb_cubes[3], '1999_liquid_water_path.nc')

#day_cubes = iris.cube.CubeList()



cube = pd_cubes[3]
reshaped_cube = cube[:,0,:,:].data
for day in range(721):
	c = cube[:, day, :, :].data
	#print cube[:, day, :,:]
	reshaped_cube = np.concatenate((reshaped_cube, c), axis = 0)

day_cubes = iris.cube.CubeList(day_cubes)
unify_time_units(day_cubes)
day_cubes.concatenate_cube()

for f in cube.slices_over('forecast_period')

for i in range(721):
