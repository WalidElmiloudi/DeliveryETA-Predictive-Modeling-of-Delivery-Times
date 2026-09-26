import pandas as pd
import numpy as np
from global_land_mask import globe
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError, ParserError

try:
    df=pd.read_csv("../data/raw/food-delivery.csv")
    print("file loaded successfully !")
except FileNotFoundError :
    print("Error: the specified file does not exists")
except EmptyDataError :
    print("Error: the file exists but contains no data")
except ParserError :
    print("Error: could not parse the file , might contains corrupted rows")
except UnicodeDecodeError :
    print("Error: Encoding issue !")
except Exception as e :
    print(f"Error: {e}")

df = df.replace(r"NaN",np.nan,regex=True)

df["Order_Date"] = pd.to_datetime(df["Order_Date"],format="%d-%m-%Y",errors="coerce")
df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"],format="%H:%M:%S",errors="coerce")
df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"],format="%H:%M:%S",errors="coerce")

df = df.astype({
    "Delivery_person_Age":"Int64",
    "Delivery_person_Ratings":"Float64",
    "multiple_deliveries":"Int64"
})

df["Road_traffic_density"] = df["Road_traffic_density"].str.strip()

df["Time_taken(min)"] = df["Time_taken(min)"].str.replace("(min) ","")
df["Time_taken(min)"] = df["Time_taken(min)"].astype("Int64")

df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ","")

df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(int(df["Delivery_person_Age"].mean()))

df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())

temp = df[df["Time_Orderd"].notna()]
difference = ((temp["Time_Order_picked"] - temp["Time_Orderd"]).dt.total_seconds()/60)%1440
average_minute_difference = int(difference.mean())
df["Time_Orderd"] = df["Time_Orderd"].fillna(df["Time_Order_picked"] - pd.to_timedelta(average_minute_difference,unit="m"))

df["Weatherconditions"] = df["Weatherconditions"].fillna(df["Weatherconditions"].mode()[0])


df = df.dropna(subset="Road_traffic_density")

df["multiple_deliveries"] = df["multiple_deliveries"].fillna(int(df["multiple_deliveries"].mean()))

df["Festival"] = df["Festival"].fillna(df["Festival"].mode()[0])

df["City"] = df["City"].fillna(df["City"].mode()[0])

for row in df.itertuples() :
    lat1 = row.Restaurant_latitude
    lon1 = row.Restaurant_longitude
    lat2 = row.Delivery_location_latitude
    lon2 = row.Delivery_location_longitude

    if globe.is_ocean(lat1,lon1):
        index = df[df["ID"] == row.ID].index
        df.drop(index=index,inplace=True)

    if globe.is_ocean(lat2,lon2):
        index = df[df["ID"] == row.ID].index
        df.drop(index=index,inplace=True)

cols = ["Restaurant_latitude","Restaurant_longitude"]
df[cols] = df[cols].abs()

df["Time_Orderd"] = df["Time_Orderd"].dt.time
df["Time_Order_picked"] = df["Time_Order_picked"].dt.time

df.to_csv("../data/processed/clean_data.csv",index=False)