import warnings
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
from statsmodels.tsa.seasonal import STL
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.utils.simulator import Simulator
import mysql.connector as sql
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,  # requires fbprophet be installed
    quadratic_model,
    sarima,
    theta,
)
from kats.utils.backtesters import BackTesterSimple
from kats.models.arima import ARIMAModel, ARIMAParams

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

db_connection = sql.connect(host='database-1.cibyktpmmxv5.eu-central-1.rds.amazonaws.com', database='zemp_dg', user='admin', password='5c0ssIwGXgfbGq6WfDMw')
db_cursor = db_connection.cursor()
db_cursor.execute('SELECT * FROM sma_accounts1')
table_rows = db_cursor.fetchall()

df = pd.DataFrame(table_rows)

print(df)
def demo_ensemble():
    air_passengers_df = pd.read_csv(
        "salesPredictionData/air_passengers.csv",
        header=0,
        names=["time", "value"],
    )
    air_passengers_df.columns = ["time", "value"]
    air_passengers_df.plot(kind='line', color='blue')
    # convert to TimeSeriesData object
    air_passengers_ts = TimeSeriesData(air_passengers_df)
    model_params = EnsembleParams(
        [
            BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
            BaseModelParams(
                "sarima",
                sarima.SARIMAParams(
                    p=2,
                    d=1,
                    q=1,
                    trend="ct",
                    seasonal_order=(1, 0, 1, 12),
                    enforce_invertibility=False,
                    enforce_stationarity=False,
                ),
            ),
            # requires fbprophet be installed
            BaseModelParams("prophet", prophet.ProphetParams()),
            BaseModelParams("linear", linear_model.LinearModelParams()),
            BaseModelParams(
                "quadratic", quadratic_model.QuadraticModelParams()),
            BaseModelParams("theta", theta.ThetaParams(m=12)),
        ]
    )

    # create `KatsEnsembleParam` with detailed configurations
    KatsEnsembleParam = {
        "models": model_params,
        "aggregation": "median",
        "seasonality_length": 12,
        "decomposition_method": "multiplicative",
    }

    # create `KatsEnsemble` model
    m = KatsEnsemble(
        data=air_passengers_ts,
        params=KatsEnsembleParam
    )

    # fit and predict
    m.fit()

    # predict for the next 30 steps
    fcst = m.predict(steps=30)

    # aggregate individual model results
    m.aggregate()

    # plot to visualize
    m.plot()


def demo():
    # take `air_passengers` data as an example
    # air_passengers_df = pd.read_csv(
    #     "salesPredictionData/air_passengers.csv",
    #     header=0,
    #     names=["time", "value"],
    # )
    air_passengers_df = series()
    air_passengers_df.columns = ["time", "value"]
    air_passengers_df.plot(kind='line', color='blue', x="time", y="value")
    # convert to TimeSeriesData object
    air_passengers_ts = TimeSeriesData(air_passengers_df)

    # prophet
    params = ProphetParams()
    m = ProphetModel(air_passengers_ts, params)
    m.fit()
    fcst = m.predict(steps=30)
    m.plot()

    print(fcst)

    # # sarima
    # warnings.simplefilter(action='ignore')
    # sarima_params = SARIMAParams(
    #     p = 2,
    #     d=1,
    #     q=1,
    #     trend = 'ct',
    #     seasonal_order=(1,0,1,12)
    #     )
    # sarima_m = SARIMAModel(data=air_passengers_ts, params=sarima_params)
    # sarima_m.fit()
    # fcst = sarima_m.predict(
    #     steps=30,
    #     freq="MS"
    #     )
    # sarima_m.plot()

    # Holt Winters
    # warnings.simplefilter(action='ignore')
    # holt_params = HoltWintersParams(
    #     trend="add",
    #     # damped=False,
    #     seasonal="mul",
    #     seasonal_periods=12,
    # )
    # holt_m = HoltWintersModel(
    #     data=air_passengers_ts,
    #     params=holt_params)

    # holt_m.fit()
    # fcst = holt_m.predict(steps=30, alpha=0.1)
    # print(fcst)
    # holt_m.plot()


def load_data():
    sales = pd.read_csv("salesPredictionData/sales_train.csv", index_col=False)

    sales_20949 = sales.loc[(sales["item_id"] == 5822)]
    # print(sales.head())
    print(sales[["item_id"]].value_counts())
    # print(sales_20949[["shop_id"]].value_counts())
    plot_20949 = pd.DataFrame()
    plot_20949["value"] = sales_20949["item_cnt_day"]
    plot_20949["date"] = pd.to_datetime(
        sales_20949["date"], format='%d.%m.%Y') - pd.to_timedelta(7, unit='d')
    plot_20949.reset_index()
    # plot_20949.set_index(['date'], inplace=True)
    plot_20949 = plot_20949.groupby(
        pd.Grouper(key="date", freq='W-MON')).agg("sum")
    plot_20949 = plot_20949.sort_values(by='date', ascending=True)
    print(plot_20949.head())
    print(plot_20949.describe())
    plot_20949.plot(kind='line', color='red')

    # show the plot
    plt.show()
    return plot_20949


def series():
    raw = pd.DataFrame()

    c = [*[x for x in range(2000001, 2000365)]]
    raw['time'] = pd.to_datetime(c, format="%Y%j")
    raw['value'] = [*[x for x in range(364)]]
    return raw


def demo_backtesting(df):
    # df = series()
    # df.set_index('time', inplace=True)
    backtester_errors = {}
    # air_passengers_df = pd.read_csv(
    #     "salesPredictionData/air_passengers.csv",
    #     header=0,
    #     names=["time", "value"],
    # )
    air_passengers_df = df
    air_passengers_df.plot(kind='line', color='blue', x="time", y="value")
    air_passengers_df.columns = ["time", "value"]
    # convert to TimeSeriesData object
    air_passengers_ts = TimeSeriesData(air_passengers_df)
    params = ARIMAParams(p=2, d=1, q=1)
    ALL_ERRORS = ['mape', 'smape', 'mae', 'mase', 'mse', 'rmse']

    backtester_arima = BackTesterSimple(
        error_methods=ALL_ERRORS,
        data=air_passengers_ts,
        params=params,
        train_percentage=75,
        test_percentage=25,
        model_class=ARIMAModel)
    backtester_arima.run_backtest()
    backtester_errors['arima'] = {}
    for error, value in backtester_arima.errors.items():
        backtester_errors['arima'][error] = value
    # additive mode gives worse results
    params_prophet = ProphetParams(seasonality_mode='multiplicative')

    backtester_prophet = BackTesterSimple(
        error_methods=ALL_ERRORS,
        data=air_passengers_ts,
        params=params_prophet,
        train_percentage=75,
        test_percentage=25,
        model_class=ProphetModel)

    backtester_prophet.run_backtest()

    backtester_errors['prophet'] = {}
    for error, value in backtester_prophet.errors.items():
        backtester_errors['prophet'][error] = value

    print(pd.DataFrame.from_dict(backtester_errors))


def change_detection(df):
    df = df.reset_index(level=0)

    df.rename(columns={'value': 'decrease', 'date': 'time'}, inplace=True)
    print(df.head())
    tsd = TimeSeriesData(df)
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
    data = seasonal_decompose(df, model="additive")
    data.plot()


def extract_features(df):
    model = TsFeatures()

    # Step 2. use .transform() method, and apply on the target time series data
    output_features = model.transform(df)
    print(output_features)


def slt_decompose(df):
    df = df.reset_index(level=0)
    # df.rename(columns={'value': 'decrease', 'date': 'time'}, inplace=True)
    stl = STL(df['value'].to_numpy(), period=7)
    res = stl.fit()
    plt.plot(
        df['date'],
        res.seasonal
    )
    plt.xticks(rotation=90)
    plt.title(
        f'Seasonal component - variance: {np.round(np.var(res.seasonal), 2)}')


def generate_ts_data():
    # simulate 90 days of data
    sim = Simulator(n=90, freq="D", start="2021-01-01")
    random_seed = 100

    # generate 10 TimeSeriesData with arima_sim
    np.random.seed(random_seed)  # setting numpy seed
    arima_sim_list = [sim.arima_sim(
        ar=[0.1, 0.05], ma=[0.04, 0.1], d=1) for _ in range(10)]

    # generate 10 TimeSeriesData with trend shifts
    trend_sim_list = [
        sim.trend_shift_sim(
            cp_arr=[30, 60, 75],
            trend_arr=[3, 15, 2, 8],
            intercept=30,
            noise=50,
            seasonal_period=7,
            seasonal_magnitude=np.random.uniform(10, 100),
            random_seed=random_seed
        ) for _ in range(10)
    ]

    # generate 10 TimeSeriesData with level shifts
    level_shift_list = [
        sim.level_shift_sim(
            cp_arr=[30, 60, 75],
            level_arr=[1.35, 1.05, 1.35, 1.2],
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=np.random.uniform(0.1, 1.0),
            random_seed=random_seed
        ) for _ in range(10)
    ]

    # for trend in trend_sim_list:
    #     trend.plot(cols=['value'])
    # for ts in level_shift_list:
    #     ts.plot(cols=['value'])
    for ts in arima_sim_list:
        ts.plot(cols=['value'])
    plt.xticks(rotation=45)
    plt.show()


def demo1():
    df_all = pd.read_csv(
        "salesPredictionData/test_dg_all.csv",
        header=0,
        names=["time", "value", "product_id"],

    )

    top_products = df_all['product_id'].value_counts()[0:10]
    for p in top_products.keys():

        df = df_all.loc[(df_all["product_id"] == p)]
        
        df = df.drop(['product_id'], axis=1)
        # print(df)
        # return
        df['time'] = pd.to_datetime(
            df['time'], format='%Y-%m-%d %H:%M:%S') - pd.to_timedelta(7, unit='d')

        df = df.groupby(
            pd.Grouper(key="time", freq='M')).agg("sum")
        df = df.sort_values(by='time', ascending=True)

        df.plot(kind='line', color='blue', y="value")

        # data = seasonal_decompose(df, model="additive")
        # data.plot()

        # ts = TimeSeriesData(df)
        df.reset_index(inplace=True)
        # demo_backtesting(df)
        # extract_features(df)

    # # prophet
    # params = ProphetParams()
    # m = ProphetModel(ts, params)
    # m.fit()
    # fcst = m.predict(steps=10)
    # m.plot()

    # print(fcst)


# demo_ensemble()
demo1()
# generate_ts_data()
# demo_backtesting()
#df = load_data()
# seasonal_decompose_data(df)
# change_detection_robuststat(df)
# extract_features(df)
# slt_decompose(df)
