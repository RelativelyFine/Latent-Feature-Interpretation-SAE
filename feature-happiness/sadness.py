import pandas as pd

emotions_dataset = pd.read_csv('feature-happiness/emotions.csv')
output_file = 'sadness.csv'
sadness_dataset = emotions_dataset[emotions_dataset["label"] == 0]