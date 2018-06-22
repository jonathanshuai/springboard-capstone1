
# Gun Violence: An Analysis of a Growing Problem in the United States

The United States has notoriously high gun violence and mass shootings incidents, and the problem seems to be getting worse over time. The primary goal of this project is to examine gun violence trends in the United States and find correlations with gun control laws. Here are some of the questions we will be answering:

* How have gun violence rates changed over the years?
* Which states have seen the biggest increases in gun violence? the lowest?
* How does gun control in states with low gun violence rates compare with gun control in states with high gun violence rates?
* Which gun control laws have the most correlation with reduced gun violence?
* Are there categories of gun control laws that perform better than others? (e.g. background checks vs. banning assault weapon?)

We are also interested to see if we can find any relationship between gun violence and features such as income, substance abuse, and other crime. Finally, we will be creating a model to help predict whether gun violence will increase in the next year for each state.

Through this project, I hope to find useful insights on gun control provisions that may help policymakers make better decisions in interest of reducing gun violence. Using data to analyze laws allows us to justify policies with evidence and learn which laws work and which ones don’t. Furthermore, by predicting gun violence increases in each state, we can take countermeasures to prevent them such as budgeting for law enforcement or implementing new gun control policies.

## Data Sources

This project compiles data from various sources for analysis with gun violence trends. Here are the sources of the data: 

* [National Institute on Alcoholism and Alcohol Abuse](https://pubs.niaaa.nih.gov/publications/surveillance110/tab4-5_16.htm)
Alcohol consumption per capita for each state from 1977-2016.

* [GunPolicy.org](http://www.gunpolicy.org/) 
Annual gun homicides data per state from 2000 to 2013.

* [Gun Violence Archive](http://www.gunviolencearchive.org/)
Gun violence incidents from January 2014 to March 2018. This dataset was downloaded from Kaggle, although it was originally scraped from the Gun Violence Archive. 

* [Disaster Center](http://www.disastercenter.com/crime/)
Annual crime factors from 2010 to 2016. The Disaster Center collected the data from the FBI UCS Annual Crime Reports.

* [US Census Bureau](https://www.census.gov/)
Two different population CSVs were downloaded from the US Census Bureau, and merged together in a simpler file ./data/raw/population.csv via a Python script merge-populations.py. This merged CSV contains annual populations for each state from 2000 to 2017.

* [Kaggle: Gun Control Provisions](https://www.kaggle.com/jboysen/state-firearms)
Gun control provisions as annual entries for each state from 1991 to 2017. This dataset was generated from several sources, including Thomson Reuters Westlaw legislative database and data from Everytown for Gun Safety and Legal Science, LLC.
Each gun provision is encoded as a shortened codename. The details about each provision and its shortened name can be found in `./data/raw/codebook.xlsx`

* [270 to Win](https://www.270towin.com/states/)
Election results from 2000 to 2016 for each state. This dataset simply lists the percentage of votes each part received in each state. 

* [Bureau of Economic Analysis](https://www.bea.gov/iTable/index_regional.cfm) 
Personal income data from 2009-2017, listed annualy and for each state.

* [Bureau of Alcohol, Tobacco, and Firearms](https://www.atf.gov/resource-center/data-statistics)
Annual gun registration data for each state from 2011-2017.

* [SAMHSA](https://www.samhsa.gov/)
Substance use for each state from survey results from 2012-2016. 6 features were selected from the survey results. They are:

All data can be found in the `/data/raw` directory. More information about each dataset can be found from the websites, or from the data wrangling document in the `/documents` folder.

## Project Structure
Here are the notebooks in the project:

1. [clean-data.ipynb](clean-data.ipynb) - Consolidation and cleaning of data into a structures easy to use for modeling and visualizations. 
2. [visualization.ipynb](visualization.ipynb) - Visualization and analysis of relationships between features.
3. [time-series.ipynb](time-series.ipynb) - Testing time series analysis.
4. [monthly-modeling.ipynb](monthly-modeling.ipynb) - Model to make predictions on gun violence increases on a monthly level.
5. [annual-modeling.ipynb](annual-modeling.ipynb) - Model to make predictions on gun violence increases on a annual level.

## Documents and Reports
Here are the documents to summarize key points and results from the notebooks:

1. [cleaning-data.pdf](/documents/cleaning-data.pdf) - Explanation of data used and the choices made in consolidating the data into a single structure.
2. [exploratory-data-analysis.pdf](/documents/exploratory-data-analysis.pdf) - Summary of the visualizations and statistical tests used in the exploratory data analysis.
3. [milestone-report.pdf](/documents/milestone-report.pdf) - Mid-project milestone report.
4. [predictive-modeling.pdf](/documents/predictive-modeling.pdf) - Summary of the results of the predictive modeling.
5. [slideshow.pdf](/documents/slideshow.pdf) - A slideshow for general audiences summarizing the key points of the project.
6. [final-report.pdf](/documents/final-report.pdf) - All of the reports aggregated into a single document.
