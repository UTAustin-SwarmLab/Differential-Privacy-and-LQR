import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


train_month_list = ["apr", "jul", "jun", "may"]
test_month_list = ["aug", "sep"]
data_direct = "./uber data/"

df = pd.DataFrame({"Date/Time":[], "Lat":[], "Lon":[], "Base":[]})
test_df = pd.DataFrame({"Date/Time":[], "Lat":[], "Lon":[], "Base":[]})


for month in train_month_list:
    df_ = pd.read_csv( data_direct + "uber-raw-data-{}14.csv".format(month) )
        # , parse_dates=["Date/Time"])
    df = pd.concat([df, df_]) # 285k rows

for month in test_month_list:
    test_df_ = pd.read_csv( data_direct + "uber-raw-data-{}14.csv".format(month) )
        # , parse_dates=["Date/Time"])
    test_df = pd.concat([test_df, test_df_]) # 285k rows


AvgLat = df["Lat"].mean()
AvgLon = df["Lon"].mean()

def DiscretizeRegion(row):
    location = 0
    if row["Lat"] >= AvgLat:
        location += 1
    if row["Lon"] >= AvgLon:
        location += 2
    
    return location

### training set: Apr - Aug
df['Date/Time'] = df['Date/Time'].astype('datetime64[ns]')
df['Region'] = df.apply(lambda row: DiscretizeRegion(row), axis=1)
df['YYYYMMDD-hh'] = df['Date/Time'].dt.strftime('%Y%m%d-%H')
# print(AvgLat, AvgLon)
df = df.groupby(['Region','YYYYMMDD-hh']).count()
df.drop(['Date/Time', "Lat", "Lon"], axis=1, inplace=True)
df.rename(columns={"Base": "Count"})
df.sort_values(by=["YYYYMMDD-hh", "Region"], inplace=True, ascending=True)
df.to_csv(data_direct + "uber training data.csv", index=True)


### testing set: Sep
test_df['Date/Time'] = test_df['Date/Time'].astype('datetime64[ns]')
test_df["Region"] = test_df.apply(lambda row: DiscretizeRegion(row), axis=1)
test_df['YYYYMMDD-hh'] = test_df['Date/Time'].dt.strftime('%Y%m%d-%H')
test_df = test_df.groupby(['Region','YYYYMMDD-hh']).count()
test_df.drop(['Date/Time', "Lat", "Lon"], axis=1, inplace=True)
test_df.rename(columns={"Base": "Count"})
test_df.sort_values(by=["YYYYMMDD-hh", "Region"], inplace=True, ascending=True)
test_df.to_csv(data_direct + "uber testing data.csv", index=True)
