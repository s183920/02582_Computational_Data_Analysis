"""Script for loading the data set and creating the wanted variables"""


import pandas as pd
import numpy as np
from pandas.tseries.holiday import *


class HolidayCalendar(AbstractHolidayCalendar):
   rules = [
     Holiday('New Year', month=1, day=1),
     Holiday('Groundhog Day', month=1, day=6),
     Holiday('St. Patricks Day', month=3, day=17),
     Holiday('April Fools Day', month=4, day=1),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Labor Day', month=5, day=1),
     Holiday('Canada Day', month=7, day=1),
     Holiday('July 4th', month=7, day=4),
     Holiday('All Saints Day', month=11, day=1),
     Holiday('Christmas', month=12, day=25),
     *USFederalHolidayCalendar.rules
   ]

class DataLoader:
    def __init__(self, path = "Realized Schedule 20210101-20220228.xlsx") -> None:
        data = pd.read_excel(path, parse_dates = True)
        data = data.sort_values(by = "ScheduleTime").reset_index(drop = True)

        # add features

        # time features
        data["Hour"] = data.ScheduleTime.dt.hour
        data["Date"] = data.ScheduleTime.dt.date
        data["Weekday"] = data.ScheduleTime.dt.day_name()
        data["Month"] = data.ScheduleTime.dt.month_name()
        data["Year"] = data.ScheduleTime.dt.year
        data["YearMonth"] = data.Year.astype(str)+  data.Month

        data["QuarterEnd"] = data.ScheduleTime.dt.to_period("Q").dt.end_time
        data["MonthEnd"] = data.ScheduleTime.dt.to_period("M").dt.end_time
        data["TimeToQuarterEnd"] = (data["QuarterEnd"] - data["ScheduleTime"]).dt.days
        data["TimeToMonthEnd"] = (data["MonthEnd"] - data["ScheduleTime"]).dt.days
        
        # holidays 
        holidays = HolidayCalendar().holidays(min(data.Date), max(data.Date)).date
        data["Holiday"] = data.Date.isin(holidays)

        # passengers
        data["Passengers"] = data.SeatCapacity*data.LoadFactor

        self.data = data

    def __add_feature(self, feature_name):
        if feature_name.lower() == "hour":
            

        

        elif feature_name == "holiday":
            

    def add_features(self, feature_names):
        feature_names = [feature_names] if isinstance(feature_names, str) else feature_names

        for feature_name in feature_names:
            self.__add_feature(feature_name)