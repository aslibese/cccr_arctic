# -*- coding: utf-8 -*-
# Name: Asli Bese

# import the functions from ArcticWarmingUtilities.py
from ArcticWarmingUtilitiesCanada import selectPeriod, find_variable_by_dims, annualMean, performLinearRegression, gridCellRegression, weightedAverage

import numpy as np  
import xarray as xr 
import argparse # to handle command line arguments
import os

parser = argparse.ArgumentParser(description='A program that requires a data file name.')
parser.add_argument('filename', type=str, help='Name of the data file')

args = parser.parse_args()

print("Processing data file:", args.filename)

# extract directory and base filename
input_directory = os.path.dirname(args.filename)
base_filename = os.path.basename(args.filename)

# remove file extension from base filename
filename_without_ext, _ = os.path.splitext(base_filename)

# open the dataset
ds = xr.open_dataset(args.filename)

# specify the start_year and end_year for the period of interest 
start_year = 1979
end_year = 2021

# extract based on the period interest
ds_with_period = selectPeriod(ds, start_year, end_year)

# define acceptable dimensions
acceptable_dims = [
	{'time', 'lat', 'lon'},
	{'time', 'latitude', 'longitude'}
]

# use the function to find the variable for temperature anomaly
var_name = find_variable_by_dims(ds_with_period, acceptable_dims)

if var_name is None:
	raise ValueError("No variable with the required dimensions found.")
else:
	print("Variable selected for processing: ", var_name)


# take the annual mean of temperature anomalies
ds_with_annualMean = annualMean(ds_with_period, var_name)

# compute the weighted average and the global warming trend
weighted_average = weightedAverage(ds_with_annualMean)
global_trend, _, gt_pvalue = performLinearRegression(weighted_average.values)

print("Global warming trend during ", str(start_year), "-", str(end_year), " is ", str(np.round(global_trend,3)), " with a p-value of ", str(np.round(gt_pvalue,3)))

# perform linear regression and compute slope and p_value for each grid cell 
ds_processed = gridCellRegression(ds_with_annualMean)

# FOR TREND
# make a copy of the dataset, this is needed to plot Arctic temperature trend
# and mask out grid cells where the p-value is statistically non-significant (>= 0.05)
ds_trend_masked = ds_processed.copy()
ds_trend_masked['slope'] = ds_trend_masked['slope'].where(ds_trend_masked['p_value'] < 0.05, np.nan)

# construct the output file path
trend_output_file_path = os.path.join(input_directory, "{}_2021_trendMasked.nc".format(filename_without_ext))

# save the processed dataset
ds_trend_masked.to_netcdf(trend_output_file_path)

print("Processed trend masked data saved to:", trend_output_file_path)


# FOR ARCTIC AMPLIFICATION
# compute the amplification in each grid cell 
ds_processed['amplification'] = ds_processed['slope'] / global_trend

# construct the output file path
aa_output_file_path = os.path.join(input_directory, "{}_2021_aa.nc".format(filename_without_ext))

# save the processed dataset
ds_processed.to_netcdf(aa_output_file_path)

print("Processed Arctic Amplification data saved to:", aa_output_file_path)

