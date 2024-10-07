#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 04 Oct 2024

This script contains the utility functions for processing Arctic warming data.

"""

import numpy as np  
import xarray as xr 
import scipy.stats as st
import pandas as pd


def fix_coords(ds):
    """
    Function to rename latitude and longitude coordinates, and convert date to datetime format and rename to time.
    """
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    if 'date' in ds.coords:
        ds['date'] = pd.to_datetime(ds['date'].astype(str), format='%Y%m%d')
        ds = ds.rename({'date': 'time'})
    return ds


def select_region(ds, lower_boundary, upper_boundary, west_boundary, east_boundary):
    """
    Function to slice the dataset based on the specificed boundaries.
    """
    lat_dim = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_dim = 'lon' if 'lon' in ds.dims else 'longitude'
    ds_region = ds.sel(**{lat_dim: slice(float(lower_boundary), float(upper_boundary)),
                          lon_dim: slice(float(west_boundary), float(east_boundary))
                          })
    return ds_region


def weighted_avg(data, lat_bound_s=-90, lat_bound_n=90, lon_bound_w=0, lon_bound_e=360):
    """
    Function to compute the spatial average while weighting for cos(latitude).
    """
    # constrain area of interest and take zonal mean
    data = data.sel(lat=slice(lat_bound_s, lat_bound_n), lon=slice(lon_bound_w, lon_bound_e))
    zonal_mean = data.mean(dim='lon')

    # compute latitude weights
    weights = np.cos(np.deg2rad(data.lat)) / np.cos(np.deg2rad(data.lat)).sum()

    # compute the weighted average across latitudes
    avg = (zonal_mean * weights).sum(dim='lat')

    return avg


def perform_linear_regression(data):
    """
    Function to perform linear regression.
    """
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = st.linregress(x, data)
    return slope, intercept, p_value


def apply_grid_cell_regression(da):
    """
    Function to perform linear regression for each grid cell and return slopes and p-values.
    """
    result = xr.apply_ufunc(
        perform_linear_regression, 
        da, 
        vectorize=True, 
        dask='parallelized', 
        input_core_dims=[['year']], 
        output_core_dims=[[], [], []], 
        output_dtypes=['float64', 'float64', 'float64']
    )

    slope = result[0]
    p_value = result[2]
    return slope, p_value