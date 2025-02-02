import pandas as pd

df = pd.read_csv("emotions.csv")

df["label"] = pd.to_numeric(df["label"])

filtered_df = df[df["label"] != 1]

filtered_df.to_csv("Not_happiness.csv",index = False)



