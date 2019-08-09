# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import time
import matplotlib.pyplot as plt
import os


def get_daily(bdt, edt):
    """

    :param bdt:
    :param edt:
    :return:
    """
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange='DCE', start_date=bdt, end_date=edt, is_open=1)
    dt = cal.loc[0]['cal_date']
    print(dt)
    mkt_prc = pro.fut_daily(exchange='DCE', trade_date=dt)
    for i in range(1, len(cal)):
        dt = cal.loc[i]['cal_date']
        print(dt)
        mkt_prc = mkt_prc.append(pro.fut_daily(exchange='DCE', trade_date=dt))
        time.sleep(0.51)

    return mkt_prc


def df2file(df, filename):
    df.to_csv(filename)


def file2df(filename, cols):
    return pd.read_csv(filename, usecols=cols)


def main_market(df, code):
    """
    品种主力合约行情
    :param df:
    :param code:
    :return:
    """


def weight_market(df, code):
    """
    品种持仓量加权行情
    :param df:
    :param code:
    :return:
    """


if __name__ == '__main__':
    fn = 'future_daily.csv'
    # df2file(get_daily('20170101', '20190731'), fn)

    df = file2df(fn, ['ts_code', 'trade_date', 'settle', 'vol', 'oi'])

    mmkt = main_market(df, 'a')
    wmkt = weight_market(df, 'a')
    fig = plt.figure(figsize=(10, 6))
    plt.plot(mmkt, label='a main', color='blue')
    plt.plot(wmkt, label='a wght', color='red')
    plt.legend()
    plt.show()
