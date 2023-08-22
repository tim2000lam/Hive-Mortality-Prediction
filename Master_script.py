import os

# Call Add_weather.py using os
os.system('python3 Add_weather.py')

# Call data_preprocessing.py using os
os.system('python3 data_preprocessing.py')

# Call Trim_data.py to remove 3/4 of the rows in the dataframe. This is optional and for reducing computational complexity of model training/testing
os.system('python3 Trim_data.py')

# Call feature_selection.py using os
os.system('python3 feature_selection.py')

# Call DT_model_Optimized.py using os
os.system('python3 DT_model_Optimized.py')

# Call GBR_model_Optimized.py using os
os.system('python3 GBR_model_Optimized.py')

# Call RF_model_Optimized.py using os
os.system('python3 RF_model_Optimized.py')

# Call SVM_model_Optimized.py using os
os.system('python3 SVM_model_Optimized.py')
