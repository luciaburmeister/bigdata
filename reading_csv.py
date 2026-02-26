import pandas as pd
import kagglehub

df_seattle = pd.read_csv("seattle_data.csv")
print("columns of seattle data:", df_seattle.columns)

df_chicago = pd.read_csv("chicago_data.csv")
print("columns of chicago data:", df_chicago.columns)

df_chicago = df_chicago.rename(columns={
    "Case Number": "Report Number",
    "Date": "Report DateTime",
    "ID": "Offense ID",
    "Primary Type": "Offense Category",
    "Description": "Offense Description",
    "Block": "Block Address"
})

df_chicago["City"] = "Chicago"