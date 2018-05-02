# Numpy and pandas for manipulating the data
import numpy as np
import pandas as pd
import scipy.stats

# Matplotlib and seaborn for visualization
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import plotly
import plotly.plotly as py
import plotly.figure_factory as ff

plotly.tools.set_credentials_file(username='jshuai', api_key='cYnSXW3VDie6Eb45YP5t')


# First let's read in our clean data
provisions_file = './data/raw/provisions.csv' # We'll need this
lat_long_file = './data/cleaned/lat_long.csv'
annual_file = './data/cleaned/annual_gun.csv'
by_date_norm_file = './data/cleaned/by_date_norm.csv'
by_date_total_file = './data/cleaned/by_date_total.csv'
feature_file = './data/cleaned/features.csv'
overall_file = './data/cleaned/overall.csv'

provisions_df = pd.read_csv(provisions_file, parse_dates=True)
lat_long_df = pd.read_csv(lat_long_file, parse_dates=True, index_col=0)
annual_df = pd.read_csv(annual_file, parse_dates=True, index_col=0)
by_date_norm_df = pd.read_csv(by_date_norm_file, parse_dates=True, index_col=0)
by_date_total_df = pd.read_csv(by_date_total_file, parse_dates=True, index_col=0)
feature_df = pd.read_csv(feature_file, parse_dates=True, index_col=0)
overall_df = pd.read_csv(overall_file, parse_dates=True, index_col=0)

# For our first visualization, let's just plot the gun homicide trend in two ways:
# Daily mean with window from 2014-2017
win_size = 60
country_gun_df = by_date_total_df['2014':'2017'].sum(axis=1)
country_gun_df.rolling(window=win_size).mean().plot()
plt.title('US Daily Gun Homicides (2014-2017)')
plt.xlabel('Date')
plt.ylabel('Number of Gun Homicides (average over {} day window)'.format(win_size))
# plt.show()

# Annual from 1999-2017
country_annual_df = annual_df[['state', 'year', 'gun_deaths']].groupby('year').sum()
country_annual_df.plot(marker='.')
plt.title('US Annual Gun Homicides (1999-2017)')
plt.xlabel('Year')
plt.ylabel('Number of Gun Homicides')
plt.ylim(0)
[plt.annotate(int(y), xy=(x, y), textcoords='data') for x, y in zip(country_annual_df.index, country_annual_df['gun_deaths'])]
# plt.show()


# Plot gun deaths as they relate to other crime, income, and total gun control laws
style_kwargs = {
                's': 3,  # set marker size
                'alpha': 0.15,
                }

plt.subplot(3, 1, 1)
sns.regplot(overall_df['other_crime_norm'], overall_df['gun_deaths_norm'], style_kwargs)

plt.subplot(3, 1, 2)
sns.regplot(overall_df['income'], overall_df['gun_deaths_norm'], style_kwargs)

plt.subplot(3, 1, 3)
sns.regplot(overall_df['lawtotal'], overall_df['gun_deaths_norm'], style_kwargs)
# plt.show()


# Plot gun deaths as they relate to alcohol consumption
plt.subplot(4, 1, 1)
sns.regplot(overall_df['beer'], overall_df['gun_deaths_norm'], style_kwargs)

plt.subplot(4, 1, 2)
sns.regplot(overall_df['wine'], overall_df['gun_deaths_norm'], style_kwargs)

plt.subplot(4, 1, 3)
sns.regplot(overall_df['spirits'], overall_df['gun_deaths_norm'], style_kwargs)

plt.subplot(4, 1, 4)
sns.regplot(overall_df['all'], overall_df['gun_deaths_norm'], style_kwargs)
# plt.show()


# Threshold number of states to have a law before we examine it
threshold = 15

def compare_provisions(annual_df, provisions_df, threshold=15, year=2017):
    """Return a DataFrame containing difference in population normed gun homicides and p_values
    
    threshold   (int): the minimum number of states that have the provision for it to be included in the results.
    This prevents misleading results (e.g. laws only Hawaii has) and helps to control for sample size difference.

    year        (int): the year which we should conduct our comparisons in. We focus on one year because we used 
    an independent t-test and observations from the same state across different years are surely not independent,
    whereas observations from different states are independent.   
    """

    # Create a dataframe with 2017 entires 
    provisions = provisions_df.columns[2:-1]
    filtered_df = pd.merge(annual_df, provisions_df)
    filtered_df = filtered_df[filtered_df['year'] == 2017]

    # Remove where provisions sum less than threshold
    provisions = provisions[filtered_df[provisions].sum() > threshold]

    # Lists to keep track of gun homicides for states with and without each law along with the p values
    a_list, b_list, p_list, a_size_list = [], [], [], []

    # Find differences between states w/wo each law
    for p in provisions:
        ab = filtered_df.groupby(p)['gun_deaths_norm'].mean()
        # Append the a and b means to a_list and b_list
        a_list.append(ab[0])
        b_list.append(ab[1])

        # Get a p value using the Welch's t-test; **assumptions of t-test
        without_law = filtered_df[filtered_df[p] == 0]['gun_deaths_norm']
        with_law = filtered_df[filtered_df[p] == 1]['gun_deaths_norm']
        p_list.append(scipy.stats.ttest_ind(without_law, with_law).pvalue)

        a_size_list.append(filtered_df[filtered_df[p] == 0].shape[0])

    # Organize our results in a dataframe to make it easy to read
    results_df = pd.DataFrame()
    results_df['provision'] = provisions
    results_df['without'] = a_list
    results_df['with'] = b_list
    results_df['diff'] = results_df['without'] - results_df['with'] 
    results_df['p_value'] = p_list
    results_df['n_without'] = a_size_list
    results_df = results_df.sort_values('diff', ascending=False)

    return results_df


results = compare_provisions(annual_df, provisions_df, 15, 2017)
print(results)

# Map out the normalized crime
# Dictionary to turn state names into abbreviations
states_codes = {
    'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 'Arizona': 'AZ', 'California': 'CA', 
    'Colorado': 'CO', 'Connecticut': 'CT', 'District of Columbia': 'DC', 'Delaware': 'DE', 
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Iowa': 'IA', 'Idaho': 'ID', 
    'Illinois': 'IL', 'Indiana': 'IN', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 
    'Massachusetts': 'MA', 'Maryland': 'MD', 'Maine': 'ME', 'Michigan': 'MI', 
    'Minnesota': 'MN', 'Missouri': 'MO', 'Northern Mariana Islands': 'MP', 'Mississippi': 'MS', 
    'Montana': 'MT', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Nebraska': 'NE', 
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'Nevada': 'NV', 
    'New York': 'NY', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 
    'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA', 'Vermont': 'VT', 'Washington': 'WA', 
    'Wisconsin': 'WI', 'West Virginia': 'WV', 'Wyoming':  'WY'
}


# Create a color scale for aesthetics
colorscale = [
            [0.0, 'rgb(233, 236, 248)']
            [0.2, 'rgb(211, 216, 240)'],
            [0.4, 'rgb(188, 197, 233)'],
            [0.6, 'rgb(144, 158, 218)'],
            [0.8, 'rgb(99, 119, 203)'],
            [1.0, 'rgb(77, 100, 196)'],
            ]

'rgb(77, 100, 196)'
'rgb(99, 119, 203)'
'rgb(144, 158, 218)'
'rgb(188, 197, 233)'
'rgb(211, 216, 240)'
'rgb(233, 236, 248)'
# Mean 2000-2017 murders
annual_mean_df = annual_df.groupby('state').mean().reset_index()
locations = annual_mean_df['state'].apply(lambda x: states_codes[x])
values = annual_mean_df['gun_deaths_norm'].round()

data = [ dict(
        type='choropleth',
        colorscale = colorscale,
        autocolorscale = False,
        locations = locations,
        z = values,
        locationmode = 'USA-states',
        text = annual_mean_df['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Homicides per 100,000 people")
        ) ]

layout = dict(
        title = 'Average US Gun Homicide Rates (2000-2017)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout)

py.iplot(fig, validate=False, filename='map')



data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lat = lat_long_df['longitude'],
        lon = lat_long_df['latitude'],
        text = lat_long_df['state'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = colorscale,
            cmin = 0,
            color = df['cnt'],
            cmax = df['cnt'].max(),
            colorbar=dict(
                title="Incoming flightsFebruary 2011"
            )
        ))]

layout = dict(
        title = 'Most trafficked US airports<br>(Hover for airport names)',
        colorbar = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot(fig, filename='lat_long_map')