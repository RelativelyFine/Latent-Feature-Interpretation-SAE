import pandas as pd

# dataSet = pd.read_csv('https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset?resource=download')
dataSet = pd.read_csv("feature-happiness/emotions.csv")
output = 'happiness.csv'

happinessSearch = dataSet[dataSet[rank']==1] #finding "1" = joy/happiness 

print(happinessSearch) 
