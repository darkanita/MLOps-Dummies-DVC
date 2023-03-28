import pandas as pd
import os

# Read in the dataset
input_dir = 'data/downloaded'
output_dir = 'data/prepared'
os.makedirs(output_dir, exist_ok=True)

# read the file with the detected encoding
churn_df = pd.read_csv(input_dir + '/ChurnData.csv')

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

churn_df.to_csv(output_dir + '/ChurnData.csv', index=False)


house_df = pd.read_csv('data/raw/housepricedata.csv')
house_df.to_csv(output_dir + '/housepricedata.csv', index=False)