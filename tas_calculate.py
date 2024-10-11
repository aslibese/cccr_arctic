#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 06 Oct 2024

This script calculates the 2m air temperature (TAS) anomalies and Arctic Amplification Factor (AAF) using CMIP6 models for the Canadian Arctic region.

Code adapted from: https://doi.org/10.5281/zenodo.6965473

author: @aslibese
"""

import numpy as np
import math as m
import pandas as pd
import datetime as dt
import glob
import xarray as xr
import os
import cftime
from pathlib import Path
import xscen

print('\n--------------------------------------------------------------------------')
print('TAS and AAF calculating using CMIP6 models for the Canadian Arctic')
print('Start time: %s' % (dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')

########################## Function Definitions ##########################

def nonuniform_area_calculating(lat, lon):
    '''
    Calculate the area of each grid cell on a non-uniform grid.

    Parameters:
    lat (numpy.ndarray): 1D array of latitudes.
    lon (numpy.ndarray): 1D array of longitudes.

    Returns:
    area (numpy.ndarray): 2D array of grid cell areas.

    '''
    re = 6371e3  # Earth's radius in meters
    area = np.full((len(lat), len(lon)), np.nan)

    # Calculate area of each grid cell
    for j in range(len(lat)-1):
        for i in range(len(lon)-1):
            dlat = np.radians(lat[j+1] - lat[j])
            dlon = np.radians(lon[i+1] - lon[i])

            # Handle crossing the prime meridian
            if abs(lon[i+1] - lon[i]) > 300:
                dlon = np.radians(360 - abs(lon[i+1] - lon[i]))

            area[j+1, i+1] = abs(dlat * dlon) * re ** 2

    # Boundary conditions for poles
    dlat = np.radians(lat[1] - lat[0])
    for i in range(len(lon)-1):
        dlon = np.radians(lon[i+1] - lon[i])
        if abs(lon[i+1] - lon[i]) > 300:
            dlon = np.radians(360 - abs(lon[i+1] - lon[i]))
        area[0, i+1] = abs(dlat * dlon) * re ** 2

    # Periodic boundary condition for longitude
    area[:, 0] = area[:, -1].copy()

    return area


def area_averaged_tas(data, ca_arc_south=66.5, ca_arc_north=83.11, ca_arc_west=360.0-141.0, ca_arc_east=360.0-52.62):
    '''
    Calculate area-weighted average temperature for the Canadian Arctic region and the global domain.

    Parameters:
    data (xarray.Dataset): Input dataset containing 'tas' variable.
    ca_arc_south (float): Southern boundary of the Canadian Arctic region.
    ca_arc_north (float): Northern boundary of the Canadian Arctic region.
    ca_arc_west (float): Western boundary of the Canadian Arctic region.
    ca_arc_east (float): Eastern boundary of the Canadian Arctic region.

    Returns:
    weighted_data_region (xarray.DataArray): Area-weighted average temperature for the Canadian Arctic region.
    weighted_data_global (xarray.DataArray): Area-weighted average temperature for the global domain.

    '''
    lat = data['lat']
    lon = data['lon']

    # adjust longitudes to 0-360 degrees if necessary
    if lon.min() < 0:
        lon = ((lon + 360) % 360)
        data = data.assign_coords({'lon': lon})

    # create masks for the Canadian Arctic region
    lat_mask = (lat >= ca_arc_south) & (lat <= ca_arc_north)
    lon_mask = (lon >= ca_arc_west) & (lon <= ca_arc_east)

    # compute grid cell areas
    area = xr.DataArray(nonuniform_area_calculating(lat.values, lon.values),
                        coords={'lat': lat, 'lon': lon},
                        dims=['lat', 'lon'])

    # apply masks to area and tas data
    area_region = area.where(lat_mask & lon_mask)
    tas_region = data['tas'].where(lat_mask & lon_mask)

    # fill NaNs in area weights with zeros
    area_region = area_region.fillna(0)
    area_global = area.fillna(0)

    # mask out missing data in tas
    area_global = area_global.where(data['tas'].notnull(), 0)

    # compute weighted means
    weighted_data_region = tas_region.weighted(area_region).mean(dim=['lat', 'lon'])
    weighted_data_global = data['tas'].weighted(area_global).mean(dim=['lat', 'lon'])

    return weighted_data_region, weighted_data_global


def running_mean(data, year_interval, run_num):
    '''
    Calculate running mean over a specified year interval.

    Parameters:
    data (xarray.DataArray): Input data array.
    year_interval (list): Start and end years to limit the analysis within this time period.
    run_num (int): Number of years for the running mean.

    Returns:
    raw_group (xarray.DataArray): Original (non-smoothed) data for each month.
    running_data (xarray.DataArray): Running mean data over the specified year interval for each month.

    '''
    raw_group_list = []
    running_data_list = []
    month_numbers = []

    # group by month (all Januaries, Februaries, etc.)
    for month, month_data in data.groupby('time.month'):
        # ensure correct year coordinate
        month_data = month_data.assign_coords(year=month_data['time.year']) # add a new coordinate 'year' to the data array
        month_data = month_data.swap_dims({'time': 'year'})                 # make the year the primary dimension to apply rolling mean over the years
        month_data = month_data.drop_vars('time')                           # remove the 'time' dimension to avoid conflicts during concatenation

        # apply rolling mean over 'year' dimension
        running_month_data = month_data.rolling(year=run_num, center=True, min_periods=1).mean()

        # limit to the specified year interval (1951-2100)
        running_month_data = running_month_data.sel(year=slice(year_interval[0], year_interval[1]))

        # remove 'month' coordinate to prevent conflicts during concatenation
        month_data = month_data.reset_coords('month', drop=True)
        running_month_data = running_month_data.reset_coords('month', drop=True)

        raw_group_list.append(month_data)
        running_data_list.append(running_month_data)
        month_numbers.append(month)

    # concatenate the data back together along a new 'month' dimension
    raw_group = xr.concat(raw_group_list, dim='month')
    running_data = xr.concat(running_data_list, dim='month')

    # assign 'month' coordinate to ensure that month dimension is labeled correctly with the actual month numbers (1-12)
    raw_group = raw_group.assign_coords(month=('month', month_numbers))
    running_data = running_data.assign_coords(month=('month', month_numbers))

    return raw_group, running_data



def remove_height(da):
    '''
    Remove 'height' coordinate from the input DataArray.
    '''
    if 'height' in da.coords:
        da = da.drop_vars('height')
    return da


#############################################################################

######################### Initial Data Settings ##############################

def main():
    ca_arc_south = 66.5
    ca_arc_north = 83.11
    ca_arc_west = 360.0 - 141.0   # 219.0 degrees East
    ca_arc_east = 360.0 - 52.62   # 307.38 degrees East

    ref_year = [1951, 1980]
    run_num = 30
    year_interval = [1951, 2100]

    # 1x1 degree grid
    lat = np.arange(-89.5, 90.5, 1)  
    lon = np.arange(-179.5, 180.5, 1) 
    ds_tgt = xr.Dataset(
        coords={
            'lat': (['lat'], lat),
            'lon': (['lon'], lon)
        }
    )
    ds_tgt['lon'].attrs['standard_name'] = 'longitude'
    ds_tgt['lat'].attrs['standard_name'] = 'latitude'

    home = Path("~").expanduser()
    historical_dir = home.joinpath("data_dir", "CMIP6/cccr_arctic/historical/tas_Amon")
    ssp245_dir = home.joinpath("data_dir", "CMIP6/cccr_arctic/ssp245/tas_Amon")

    historical_files = sorted(glob.glob(os.path.join(historical_dir, '*.nc')))
    ssp_files = sorted(glob.glob(os.path.join(ssp245_dir, '*.nc')))

    model_names = []
    for f in historical_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        model_name = parts[2]
        model_names.append(model_name)

    model_names = list(set(model_names))
    model_names.sort()

    tarctic_moving_diff_list = []
    tglobal_moving_diff_list = []
    aaf_moving_list = []
    models_processed = []

    for model in model_names:
        hist_files = glob.glob(os.path.join(historical_dir, '*_%s_*.nc' % model))
        ssp_files = glob.glob(os.path.join(ssp245_dir, '*_%s_*.nc' % model))

        print('\nProcessing model:', model)
        ds_hist = xr.open_dataset(hist_files[0])
        ds_ssp = xr.open_dataset(ssp_files[0])

        # concatenate hist and ssp245 datasets along time dimension
        ds = xr.concat([ds_hist, ds_ssp], dim='time')

        # identify the time type
        time_type = type(ds['time'].values[0])
        print(f"Time type for model {model}: {time_type}")

        # set start and end dates to slice
        start_date = pd.to_datetime('1900-01-01')
        end_date = pd.to_datetime('2100-12-31')

        # slice the dataset to the specified time period handing different time formats
        try:
            if isinstance(ds['time'].values[0], cftime.datetime): # 2000-01-01 12:00:00
                # convert start and end dates to cftime objects for the dataset's calendar
                start_date = ds['time'].values[0].__class__(1900, 1, 1)
                end_date = ds['time'].values[0].__class__(2100, 12, 31)

                print("Converting time to CFTimeIndex...")
                ds = ds.sel(time=slice(start_date, end_date))

                # convert CFTimeIndex to pandas DateTimeIndex for consistency
                time_values = ds.indexes['time'].to_datetimeindex()
                ds['time'] = pd.DatetimeIndex(time_values)

            elif isinstance(ds['time'].values[0], np.datetime64): # 2000-01-01T12:00
                print("Handling numpy datetime64 time format...")
                ds = ds.sel(time=slice(start_date, end_date))

            elif isinstance(ds['time'].values[0], pd.Timestamp): # 2000-01-01 12:00:00
                # Adjust start_date and end_date if necessary
                time_min = ds['time'].values[0]
                time_max = ds['time'].values[-1]

                if start_date < time_min:
                    start_date = time_min
                if end_date > time_max:
                    end_date = time_max
                ds = ds.sel(time=slice(start_date, end_date))

            else:
                # skip the model if time format is unrecognized
                print(f"Error: Unrecognized time format {time_type} for model {model}. Skipping this model.")
                continue

            print(f"Sliced time for model {model}: {ds['time'].values}")

            # regrid to 1x1 resolution using xscen
            weights_location = home.joinpath("my_dir", "cccr_arctic/weights")
            ds_regridded = xscen.regrid.regrid_dataset(ds, ds_tgt, regridder_kwargs={"method": "conservative_normed", "skipna": True}, weights_location=weights_location)
            print(f"Regridding for model {model} completed.")  

            weighted_data_region, weighted_data_global = area_averaged_tas(ds_regridded)

            if 'month' in weighted_data_region.coords:
                weighted_data_region = weighted_data_region.drop_vars('month')
            if 'month' in weighted_data_global.coords:
                weighted_data_global = weighted_data_global.drop_vars('month')

            # extract year and month values from the time coordinate and assign them to new 'month' and 'year' coords along the time dimension
            weighted_data_region = weighted_data_region.assign_coords(
                month=('time', weighted_data_region['time'].dt.month.values),
                year=('time', weighted_data_region['time'].dt.year.values)
            )
            weighted_data_global = weighted_data_global.assign_coords(
                month=('time', weighted_data_global['time'].dt.month.values),
                year=('time', weighted_data_global['time'].dt.year.values)
            )

            # compute monthly climatology over reference period
            ref_arctic = weighted_data_region.sel(
                time=((weighted_data_region['year'] >= ref_year[0]) & (weighted_data_region['year'] <= ref_year[1]))
            ).groupby('month').mean(dim='time')
            ref_global = weighted_data_global.sel(
                time=((weighted_data_global['year'] >= ref_year[0]) & (weighted_data_global['year'] <= ref_year[1]))
            ).groupby('month').mean(dim='time')

            # running mean (30-year window over monthly data)
            tarctic_raw_group, tarctic_moving = running_mean(weighted_data_region, year_interval, run_num)
            tglobal_raw_group, tglobal_moving = running_mean(weighted_data_global, year_interval, run_num)

            # monthly anomalies: difference between the mean of a running 30 yr period and the mean of the 1951–1980 reference period
            tarctic_moving_diff = tarctic_moving.groupby('month') - ref_arctic
            tglobal_moving_diff = tglobal_moving.groupby('month') - ref_global

            # AAF : 'tas' changes in the 30 yr mean in the Arctic divided by the 30 yr mean in the global domain
            aaf_moving = tarctic_moving_diff / tglobal_moving_diff

            # expand dimensions to include 'model'
            tarctic_moving_diff = tarctic_moving_diff.expand_dims('model')
            tglobal_moving_diff = tglobal_moving_diff.expand_dims('model')
            aaf_moving = aaf_moving.expand_dims('model')

            tarctic_moving_diff = tarctic_moving_diff.assign_coords(model=[model])
            tglobal_moving_diff = tglobal_moving_diff.assign_coords(model=[model])
            aaf_moving = aaf_moving.assign_coords(model=[model])

            tarctic_moving_diff_list.append(tarctic_moving_diff)
            tglobal_moving_diff_list.append(tglobal_moving_diff)
            aaf_moving_list.append(aaf_moving)
            models_processed.append(model)

        except Exception as e:
            print(f"An error occurred while processing model {model}: {e}")
            print("Skipping this model and continuing with the next one.")
            continue
    
    
    tarctic_moving_diff_list = [remove_height(da) for da in tarctic_moving_diff_list]
    tglobal_moving_diff_list = [remove_height(da) for da in tglobal_moving_diff_list]
    aaf_moving_list = [remove_height(da) for da in aaf_moving_list]

    tarctic_all = xr.concat(tarctic_moving_diff_list, dim='model')
    tglobal_all = xr.concat(tglobal_moving_diff_list, dim='model')
    aaf_all = xr.concat(aaf_moving_list, dim='model')

    # calculate the MMM
    MMM_tarctic = tarctic_all.mean(dim='model')
    MMM_tglobal = tglobal_all.mean(dim='model')
    MMM_aaf = aaf_all.mean(dim='model')

    output_ds = xr.Dataset({
        'tarctic_moving_diff': tarctic_all,
        'tglobal_moving_diff': tglobal_all,
        'aaf_moving': aaf_all,
        'MMM_tarctic': MMM_tarctic,
        'MMM_tglobal': MMM_tglobal,
        'MMM_aaf': MMM_aaf
    })

    output_filename = home.joinpath("data_dir", "CMIP6/cccr_arctic/tas_cmip6_month_v4.nc")
    output_ds.to_netcdf(output_filename)
    print('\nOutput saved to', output_filename)

    print(f'\n {len(models_processed)} models processed: {models_processed}')


if __name__ == '__main__':
    main()

print('\n--------------------------------------------------------------------------')
print('Finish time: %s' % (dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')
