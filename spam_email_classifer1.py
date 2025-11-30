import pandas as pd
import numpy as np


df=pd.read_csv("/Users/lothariandawnstar/Desktop/enron_spam_data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)


df["label"]=df["Spam/Ham"].map({"spam":1,"ham":0})
df["text"]=df["Message"].fillna("")+ " " +df["Subject"].fillna("")
df["text"]=df["text"].str.lower()
df=df[df["text"].str.strip().str.len()>0]
df=df[df["label"].isin([0,1])]
df=df.drop(columns=["Spam/Ham","Message","Subject","Unnamed: 0","Date"])
df=df.reset_index(drop=True)

print(df.head())
print(df.info())
df=df.head(5000)
df.to_csv("clean_spam_data.csv", index=False)

