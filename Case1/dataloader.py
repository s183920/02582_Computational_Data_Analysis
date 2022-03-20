import pandas as pd
import numpy as np
from pandas.tseries.holiday import *
import copy
# from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


class HolidayCalendar(AbstractHolidayCalendar):
   rules = [
     Holiday('New Year', month=1, day=1),
     Holiday('Groundhog Day', month=1, day=6),
     Holiday('St. Patricks Day', month=3, day=17),
    #  Holiday('April Fools Day', month=4, day=1),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Labor Day', month=5, day=1),
     Holiday('Canada Day', month=7, day=1),
     Holiday('July 4th', month=7, day=4),
     Holiday('All Saints Day', month=11, day=1),
     Holiday('Christmas', month=12, day=25),
     *USFederalHolidayCalendar.rules
   ]

class DataLoader:
    """Data loader"""
    def __init__(self, data = "Realized Schedule 20210101-20220228.xlsx") -> None:
      self.data = self.get_data(data) if isinstance(data, str) else data

    def get_data(self, path = "Realized Schedule 20210101-20220228.xlsx"):
        self.path = path

        data = pd.read_excel(path, parse_dates = True)
        data = data.sort_values(by = "ScheduleTime").reset_index(drop = True)
        data = data[data["LoadFactor"] > 0]

        # add features
        data["FlightNumber"] = data.Airline+data.FlightNumber.astype("str") # add airloine to flight number to make sure flight number is not duplicated between airlines

        # time features
        data["Hour"] = data.ScheduleTime.dt.hour
        data["Date"] = data.ScheduleTime.dt.date
        data["Weekday"] = data.ScheduleTime.dt.day_name()
        data["Month"] = data.ScheduleTime.dt.month_name()
        data["Year"] = data.ScheduleTime.dt.year
        data["YearMonth"] = data.Year.astype(str)+  data.Month
        data["DayMonth"] = data.ScheduleTime.dt.date.apply(lambda x: datetime.strftime(x, "%d-%m"))

        data["QuarterEnd"] = data.ScheduleTime.dt.to_period("Q").dt.end_time
        data["MonthEnd"] = data.ScheduleTime.dt.to_period("M").dt.end_time
        data["TimeToQuarterEnd"] = (data["QuarterEnd"] - data["ScheduleTime"]).dt.days
        data["TimeToMonthEnd"] = (data["MonthEnd"] - data["ScheduleTime"]).dt.days
        
        # holidays 
        holidays = HolidayCalendar().holidays(min(data.Date), max(data.Date)).date
        data["Holiday"] = data.Date.isin(holidays)

        # passengers
        data["Passengers"] = data.SeatCapacity*data.LoadFactor

        # change data types
        data.Hour = data.Hour.astype(str)
        data.Year = data.Year.astype(str)
        data.FlightNumber = data.FlightNumber.astype(str)
        data.AircraftType = data.AircraftType.astype(str)

        self.data = data

        return self.data

    def special_transforms(self):
      ## Encode time of day as cos and sin
      seconds_from_midnight = ((self.data.ScheduleTime - self.data.ScheduleTime.dt.normalize()) / pd.Timedelta('1 second')).astype(int)
      radians= 2*np.pi*seconds_from_midnight/(3600*24)
      self.data["TimeCos"] = radians.apply(np.cos)
      self.data["TimeSin"] = radians.apply(np.sin)

      ## Encode day of month as cos + sin
      fraction_of_month = (self.data.ScheduleTime.dt.day-1) / (self.data.MonthEnd.dt.day-1)
      radians=2*np.pi*fraction_of_month
      self.data["DayCos"] = radians.apply(np.cos)
      self.data["DaySin"] = radians.apply(np.sin)

      ## Encode month as cos + sin
      fraction_of_year = (self.data.ScheduleTime.dt.month-1) / 11
      radians=2*np.pi*fraction_of_year
      self.data["MonthCos"] = radians.apply(np.cos)
      self.data["MonthSin"] = radians.apply(np.sin)

      return self

    def get_subset(self, *features):
      """Get a data loader with only columns given by feature_names in the data"""
      default_features = ['ScheduleTime', 'Airline', 'FlightNumber', 'Destination',
       'AircraftType', 'FlightType', 'Sector', 'SeatCapacity', 'LoadFactor',
       'Hour', 'Weekday', 'Month', 'Year', 'TimeToQuarterEnd', 'TimeToMonthEnd', 'Holiday',
       'Passengers']
      features = list(features) if features else list(self.data.columns[self.data.dtypes == object])
      if features == ["all"]:
        return self

      # feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
      _data_loader = copy.copy(self)
      _data_loader.data = _data_loader.data[features]

      return _data_loader

    def get_possible_features(self):
      print("Possible feature:")
      print(self.data.columns)
      return self.data.columns

    def get_split(self, split_date = "2022-02-01"):
      # in time series we always split at a set date
      test = DataLoader(self.data[self.data["ScheduleTime"] >= split_date])
      train = DataLoader(self.data[self.data["ScheduleTime"] < split_date])
      return train, test

    def __onehot_encode(self, feature):
      """One hot encode the feature"""
      # #creating instance of one-hot-encoder
      # encoder = OneHotEncoder(handle_unknown='ignore')

      # #perform one-hot encoding on 'feature' column 
      # encoder_df = pd.DataFrame(encoder.fit_transform(self.data[[feature]]).toarray())

      # # set interpretable column names
      # encoder_df.columns = [feature + "_" + name for name in np.unique(self.data[[feature]]).astype(str)]

      # # join with original data
      # self.data = self.data.join(encoder_df)
      # self.data.drop(feature, axis=1, inplace=True)

      # get one hot encoding
      onehot_data = pd.get_dummies(self.data[feature])

      # set column names
      onehot_data.columns = [feature + "_" + name for name in np.unique(self.data[feature]).astype(str)]

      # join with original data
      self.data = self.data.join(onehot_data)
      self.data.drop(feature, axis=1, inplace=True)
    
    def onehot_encode(self, *features):
      features = list(features) if features else list(self.data.columns[self.data.dtypes == object])

      for feature in features:
        self.__onehot_encode(feature)
      
      return self

    def __standardize(self, feature, train_data = None):
      x = self.data[feature] if train_data is None else train_data[feature]
      data = copy.deepcopy(self.data)
      self.standardize_params = [x.mean(),x.std()] 
      data.loc[:, feature] = (self.data[feature]-self.standardize_params[0])/self.standardize_params[1]
      self.data = data

    def standardize(self, train_data = None, *features):
      features = list(features) if features else list(self.data.columns[self.data.dtypes.isin([np.float64, np.number, np.int64])])

      for feature in features:
        self.__standardize(feature, train_data=train_data)
      
      return self