from pydantic import BaseModel

class Features(BaseModel):
    Delivery_person_Age: int
    Delivery_person_Ratings: float
    Vehicle_condition: int
    multiple_deliveries: int
    Distance_km: float
    Order_picked_hour: int
    Pickup_delay_minute: int
    Weatherconditions: str
    Road_traffic_density: str
    Type_of_vehicle: str
    Festival: str
    City: str
