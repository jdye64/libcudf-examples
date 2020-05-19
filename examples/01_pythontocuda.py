import cudf

# Cython bindings for the custom CUDA kernel
import cudfkernel

print("Create a cuDF Dataframe using the Python library")

# CSV reader options
column_names = ["station_id", "date", "type", "val", "m_flag", "q_flag", "s_flag", "obs_time"]
usecols = column_names[0:4]

# All 2010 weather recordings
weather_df = cudf.read_csv("../data/weather/2010.csv.gz", names=column_names, usecols=usecols)
print("DataTypes: " + str(weather_df.dtypes))
print(weather_df.head())

# There are 5 possible recording types. PRCP, SNOW, SNWD, TMAX, TMIN
# Rainfall is stored as 1/10ths of MM.
rainfall_df = weather_df['type'] == 'PRCP'

# Snowfall is stored as MMs.
snowfall_df = weather_df['type'] == 'SNOW'

# Snowdepth is stored as MMs
snowdepth_df = weather_df['type'] == 'SNWD'

# TMAX - Max temperature. 1/10ths of degree celcius
tmax_df = weather_df['type'] == 'TMAX'

# TMIN - Min temperature. 1/10ths of degree celcius
tmin_df = weather_df['type'] == 'TMIN'

# Run the custom Kernel on the specified Dataframe Columns
rainfall_kernel = cudfkernel.CudfWrapper(weather_df)
rainfall_kernel.tenth_mm_to_inches(3)
print("after this???")
#print(weather_df.head())