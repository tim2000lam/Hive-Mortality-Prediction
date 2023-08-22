import pandas as pd 
#import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

#Read the CSV file using pandas
bee_data = pd.read_csv("bee_data_weather.csv")

#Showthe first 5 rows of the DataFrame to ensure it loaded properly
#print(bee_data.head())


#Convert the activity column to numerical values
bee_data['Activity'] = bee_data['Activity'].replace({
    'Hobby': 1,
    'Professional': 2,
    'Part_time': 3
})


country_mapping = {'LATVIA': 0, 'GREECE': 1, 'SLOVAKIA': 2, 'POLAND': 3, 'ESTONIA': 4, 'ENGLAND & WALES': 5, 'FINLAND': 6, 'LITHUANIA': 7, 'SWEDEN': 8, 'ITALY': 9, 'GERMANY': 10, 'FRANCE': 11, 'HUNGARY': 12, 'DENMARK': 13, 'BELGIUM': 14, 'SPAIN': 15}
bee_data['Country'] = bee_data['Country'].map(country_mapping)



bee_data['Breed'] = bee_data['Breed'].replace({
    'A. m. mellifera': 1,
    'A. m. ccm': 2,
    'A. m. carnica': 3,
    'Hybrid': 4,
    'Buckfast': 5,
    'Local bees': 6,
    'A. m. ligustica': 7,
    'A. m. iberiensis': 8
})


bee_data['Management'] = bee_data['Management'].replace({
    'Livestock': 1,
    'Production + Livestock': 2,
    'No Management': 3,
    'Production + Livestock  + HealthConditions': 4,
    'Livestock  + HealthConditions': 5,
    'Production': 6,
    'Production + HealthConditions': 7,
    'HealthConditions': 8
})




#Convert environment to numerical
bee_data['Environment'] = bee_data['Environment'].replace({
    'Diverse': 1,
    'Farmland': 2,
    'Wood': 3,
    'Flora': 4,
    'Orchards': 5,
    'Town': 6
})



bee_data['Program'] = bee_data['Program'].apply(lambda x: 1 if x == 'Second Year' else 0)



#make a function that only takes the columns with numbers at the beginning of the string and uses only the number
def clean_data(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: int(re.findall(r'\d+', x)[0]) if str(x)[0].isdigit() else x)
    return df

clean_data(bee_data,['Age','Beekeep_for','Production','Bee_population_size','Apiary_Size','Swarm_bought','Swarm_produced','Queen_bought','Queen_produced','Winter_Mortality_Class'])


#Define a function to convert suffering columns to 1 if they are suffering and 0 to not suffering 
def convert_suffering_columns(df):
    #Find a list of columns with suffering or not suffering data
    suffering_cols = [col for col in df.columns if set(df[col].unique()) == {'Suffering', 'Not_Suffering'}]
    #Change each column to numerical values
    for col in suffering_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Suffering' else 0)
    return(df)

convert_suffering_columns(bee_data)


def convert_true_columns(df):
    #Find a list of columns with true or false values
    categ_cols = [col for col in df.columns if set(df[col].unique()) == {'Yes', 'No'}]

    #Change each column to numerical values
    for col in categ_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    return(df)


convert_true_columns(bee_data)



#Drop ID_api column and program column:
bee_data = bee_data.drop(columns=['ID_api'])
bee_data = bee_data.drop(columns=['Program'])


#Drop the weather data column to check for non-integer values in other columns:
bee_data_dropped_weather = bee_data.drop(columns =["Mean_Temperature"])

#Loop through every cell, ensuring all values are integers.
list_NAs = []
for i, row in bee_data_dropped_weather.iterrows():
    for j, val in row.items():
        if not isinstance(val, int):
            list_NAs.append(i)

#Any values that are not integers at this point are NAs or None and will be removed

bee_data = bee_data.drop(list_NAs)


#Now, ensure all values are integers (except in weather column) 
#Drop the weather data column to check for non-integer values in other columns:
bee_data_dropped_weather = bee_data.drop(columns =["Mean_Temperature"])


#Loop through every cell, ensuring all values are integers.
list_NAs = []
for i, row in bee_data_dropped_weather.iterrows():
    for j, val in row.items():
        if not isinstance(val, int):
            print(f"Value {val} at index ({i}, {j}) is not an integer.")
            

bee_data.to_csv("bee_data_preprocessed.csv", header = True, index = False)
