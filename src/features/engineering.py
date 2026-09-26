import math
import pandas as pd
import numpy as py

df = pd.read_csv("../data/processed/clean_data.csv")

def haversin_distance(lat1,lon1,lat2,lon2):
    R = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi/2)**2) + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a),math.sqrt(1 - a))
    d = R * c

    return d

distances = []

for row in df.itertuples():

    lat1 = row.Restaurant_latitude
    lon1 = row.Restaurant_longitude
    lat2 = row.Delivery_location_latitude
    lon2 = row.Delivery_location_longitude

    distances.append(round(haversin_distance(lat1,lon1,lat2,lon2),1))

df["Distance_km"] = distances

df["Order_Date"] = pd.to_datetime(df["Order_Date"],format="%d-%m-%Y",errors="coerce")
df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"],format="%H:%M:%S",errors="coerce")
df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"],format="%H:%M:%S",errors="coerce")

differences = ((df["Time_Order_picked"] - df["Time_Orderd"]).dt.total_seconds()/60)%1440

df["Order_picked_hour"] = df["Time_Order_picked"].dt.hour
df["Pickup_delay_minute"] = differences

df = df.drop(columns=["ID","Delivery_person_ID","Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude","Order_Date","Time_Orderd","Time_Order_picked","Type_of_order"])

df.to_csv("../data/processed/featured.csv",index=False)