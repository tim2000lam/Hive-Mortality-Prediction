import numpy as np
import pandas as pd


weather_df = pd.read_csv("weather_data.csv")


radiation_cols = weather_df.columns[weather_df.columns.str.contains('radiation')]
weather_df = weather_df.drop(radiation_cols, axis=1)

#print(weather_df)


weather_start = weather_df.loc[weather_df['utc_timestamp'] == '2012-09-01T00:00:00Z'].index[0]

weather_end = weather_df.loc[weather_df['utc_timestamp'] == '2014-08-31T23:00:00Z'].index[0]

weather_df = weather_df.loc[weather_start:weather_end]


#Now, our weather df contains only temperature data from between autumn 2012 and summer 2013. 
#Next, extract the weather information for the 16 countries of interest.

countries_dict = {'BE':'BELGIUM','DK':'DENMARK','GB':'ENGLAND & WALES','EE':'ESTONIA','FI':'FINLAND','FR':'FRANCE','DE':'GERMANY','GR':'GREECE','HU':'HUNGARY','IT':'ITALY','LV':'LATVIA','LT':'LITHUANIA','PL':'POLAND','SK':'SLOVAKIA','ES':'SPAIN','SE':'SWEDEN'}



# Drop any rows with missing values
weather_df = weather_df.dropna()


for country in countries_dict:
    if country +"_temperature" in weather_df.columns:
        # Check for outliers less than -40 or more than 50
        outlier_indices = weather_df[(weather_df[country + "_temperature"] < -40) | (weather_df[country + "_temperature"] > 50)].index
        # Replace outliers with the previous value
        weather_df[country + "_temperature"].iloc[outlier_indices] = weather_df[country + "_temperature"].shift(1).iloc[outlier_indices]
        # Calculate the mean temperature for the country
        mean_temp = weather_df[country+"_temperature"].mean()
        countries_dict[country] = (countries_dict[country], mean_temp)



apiary_df = pd.read_csv("883eax1-sup-0001.csv")
apiary_df.loc[:,["Mean_Temperature"]] = [0]




for index, row in apiary_df.iterrows():
    country = row['Country']
    #print(country)
    for ct in countries_dict:
        #print(countries_dict[ct])
        if countries_dict[ct][0] == country:  #conditional to check if item of value 0 in countries dict matches row in country 
            #print("match")
            apiary_df.loc[apiary_df['Country'] == country, 'Mean_Temperature'] = countries_dict[ct][1] #If it does, replace the value in Mean_Temperature with value 1 in countries dict (the mean temperature) 

apiary_df.to_csv("bee_data_weather.csv", index=False)







        
