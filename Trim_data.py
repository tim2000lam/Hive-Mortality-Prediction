import pandas as pd
import numpy as np

# Read the original DataFrame from a CSV file
df = pd.read_csv('bee_data_preprocessed.csv')

# Calculate the number of rows to keep
num_rows = len(df) // 4

# Randomly select rows to keep
indices = np.random.choice(df.index, size=num_rows, replace=False)
df_trimmed = df.loc[indices]

# Save the trimmed DataFrame to a new CSV file
df_trimmed.to_csv('bee_data_preprocessed.csv', index=False)
