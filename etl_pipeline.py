import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')
# Remove duplicates
messages = messages[~messages.duplicated(subset='id')]

# load categories dataset
categories = pd.read_csv('categories.csv')
# Remove duplicates
categories = categories[~categories.duplicated(subset='id')]

# Split categories into 36 separate columns
cat_split = categories['categories'].str.split(';', expand=True)
# Rename the categories columns; use the first row to infer the names
cat_split.columns = cat_split.iloc[0,:].apply(lambda x : x[:-2])
# Convert the entries to 0/1
cat_split = cat_split.applymap(lambda x : x[-1]).astype('int')
# Bring back the 'id' column (will be joined on index)
categories = cat_split.join(categories['id'])

# merge datasets
df = pd.merge(messages, categories, on='id', how='outer', validate='1:1')

# Save the cleaned merged dataset to an SQLite database
engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('merged', engine, index=False)
