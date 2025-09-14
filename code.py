import  pandas as pd
from sklearn.linear_model import LinearRegression 
df=pd.read_csv("Housing.csv")
target=df["price"]
feature= df["area","bedrooms"]
reg=LinearRegression()
reg.fit(feature,target)
reg.predict(4,2309)
