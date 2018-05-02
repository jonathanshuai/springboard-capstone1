# Numpy and pandas for manipulating the data
import numpy as np
import pandas as pd

# Read in the data 
data_2000 = './data/raw/population_2000_2010.csv'
data_2010 = './data/raw/population_2010_2017.csv'

df_2000 = pd.read_csv(data_2000)
df_2010 = pd.read_csv(data_2010)

# Lowercase the column names
df_2000.columns = [x.lower() for x in df_2000.columns]
# filter for the total population of each state and drop useless columns
total_filter = (df_2000[['sex', 'origin', 'race', 'agegrp']] == 0).all(axis=1)
df_2000 = df_2000[total_filter].drop(['estimatesbase2000', 'popestimate2010', 'sex', 'origin', 'race', 'agegrp', 'region', 'division', 'state'], axis=1)
# rename the columns and reindex with state
df_2000.columns = ['state', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010']
df_2000 = df_2000.set_index('state')

# Rename the columns and get only the useful ones
df_2010.columns = ['Id', 'Id2', 'Geography', 'april2010', 'april2010_est', '2010',
                        '2011', '2012', '2013', '2014', '2015', '2016', '2017']
df_2010 = df_2010[['Geography', '2011', '2012', '2013', '2014', '2015', '2016', '2017']]

# Index by the state
df_2010 = df_2010.set_index('Geography')
df_2010 = df_2010.loc['Alabama':, :].astype(int) # Slice only the state rows

# concat the 2000 and 2010 dataframes and save it
population_df = pd.concat([df_2000, df_2010], axis=1).dropna()
population_df = population_df.astype(int)
population_df.to_csv('./data/raw/population.csv')