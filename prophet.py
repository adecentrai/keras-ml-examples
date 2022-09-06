import pandas as pd
from matplotlib import pyplot as plt, colors as clr
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams
import numpy as np
from kats.consts import TimeSeriesData
from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.robust_stat_detection import RobustStatDetector
from statsmodels.tsa.seasonal import seasonal_decompose
from kats.tsfeatures.tsfeatures import TsFeatures
def demo():
    # take `air_passengers` data as an example
    air_passengers_df = pd.read_csv(
        "salesPredictionData/air_passengers.csv",
        header=0,
        names=["time", "passengers"],
    )

    # convert to TimeSeriesData object
    air_passengers_ts = TimeSeriesData(air_passengers_df)

    # create a model param instance
    # additive mode gives worse results
    params = ProphetParams(seasonality_mode='multiplicative')

    # create a prophet model instance
    m = ProphetModel(air_passengers_ts, params)

    # fit model simply by calling m.fit()
    m.fit()

    # make prediction for next 30 month
    fcst = m.predict(steps=30, freq="MS")
    fcst.plot()
    print(fcst)


def load_data():
    sales = pd.read_csv("salesPredictionData/sales_train.csv", index_col=False)
    
    sales_20949 = sales.loc[(sales["item_id"] == 5822) ]
    # print(sales.head())
    print(sales[["item_id"]].value_counts())
    # print(sales_20949[["shop_id"]].value_counts())
    plot_20949 = pd.DataFrame()
    plot_20949["value"] = sales_20949["item_cnt_day"]
    plot_20949["date"] = pd.to_datetime(sales_20949["date"], format='%d.%m.%Y') -  pd.to_timedelta(7, unit='d')
    plot_20949.reset_index()
    # plot_20949.set_index(['date'], inplace=True)
    plot_20949 = plot_20949.groupby(pd.Grouper(key="date", freq='W-MON')).agg("sum")
    plot_20949 = plot_20949.sort_values(by='date', ascending=True)
    print(plot_20949.head())
    print(plot_20949.describe())
    plot_20949.plot(kind='line', color='red')

    # show the plot
    plt.show()
    return plot_20949


def change_detection(df):
    df = df.reset_index(level=0)

    df.rename(columns={'value': 'decrease', 'date': 'time'}, inplace=True)
    print(df.head())
    tsd = TimeSeriesData(df)
    # np.random.seed(10)
    # df_increase_decrease = pd.DataFrame(
    #     {
    #         'time': pd.date_range('2019-01-01', periods=60),
    #         'increase':np.concatenate([np.random.normal(1,0.2,30), np.random.normal(2,0.2,30)]),
    #         'decrease':np.concatenate([np.random.normal(1,0.3,50), np.random.normal(0.5,0.3,10)]),
    #     }
    # )
    # print(df_increase_decrease.head())
    # tsd = TimeSeriesData(df_increase_decrease.loc[:,['time','increase']])
    detector = CUSUMDetector(tsd)

    change_points = detector.detector(
        change_direction=["decrease"], start_point=0)
    detector.plot(change_points)
    plt.xticks(rotation=45)
    plt.show()

def change_detection_robuststat(df):
    df = df.reset_index(level=0)
    df.rename(columns={'value': 'decrease', 'date': 'time'}, inplace=True)    
    tsd = TimeSeriesData(df)
    detector = RobustStatDetector(tsd)
    change_points = detector.detector()

    detector.plot(change_points)
    plt.xticks(rotation=45)
    plt.show()

def seasonal_decompose_data(df):
    data = seasonal_decompose(df,model="additive")
    data.plot()

def extract_features(df):
    model = TsFeatures()

    # Step 2. use .transform() method, and apply on the target time series data
    output_features = model.transform(df)
    print(output_features)

df = load_data()
seasonal_decompose_data(df)
change_detection_robuststat(df)
extract_features(df)
