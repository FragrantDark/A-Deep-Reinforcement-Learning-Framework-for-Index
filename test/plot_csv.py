# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import time
import matplotlib.pyplot as plt


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


def tsc_market(df, code):
    """
    1. 品种主力合约行情
    2. 品种持仓量加权行情
    :param df:
    :param code:
    :return:
    """
    code = code.upper()
    ddic = {}   # ddic[date] = [(code, price, oi)]
    for i in range(len(df)):
        v = df.loc[i]
        tsc = v['ts_code']
        if v['vol'] == 0 or tsc.find(code) != 0 or len(tsc)-len(code) < 8:
            continue
        dt = v['trade_date']
        print('%s\t%s' % (dt, tsc))
        if dt not in ddic:
            ddic[dt] = []
        ddic[dt].append((tsc, float(v['settle']), float(v['oi'])))

    rr = []     # (date, main_prc, weight_prc, main_code)
    for dt in ddic:
        mx_oi, mx_prc, mx_code = 0, 0.0, ''
        sm_oi, sm_prc = 0, 0.0
        for t in ddic[dt]:
            sm_oi += t[2]
            sm_prc += t[2] * t[1]
            if t[2] > mx_oi:
                mx_oi = t[2]
                mx_prc = t[1]
                mx_code = t[0]
        rr.append((dt, mx_prc, sm_prc/sm_oi, mx_code))

    rr = sorted(rr, key=lambda x: x[0])
    print(rr)

    rng = []    # (b, e, main_code)
    bpos = 0
    for i in range(1, len(rr)):
        if rr[i][-1] != rr[i-1][-1]:
            rng.append((bpos, i, rr[i-1][-1]))
            bpos = i
    rng.append((bpos, len(rr), rr[-1][-1]))
    print(rng)

    main_mkt = [x[1] for x in rr]
    mres = []   # (x, y, code)
    for t in rng:
        mres.append((list(range(t[0], t[1])), main_mkt[t[0]:t[1]], t[2]))

    return mres, [x[2] for x in rr]


if __name__ == '__main__':
    fn = 'future_daily.csv'
    # df2file(get_daily('20170101', '20190731'), fn)

    df = file2df(fn, ['ts_code', 'trade_date', 'settle', 'vol', 'oi'])

    code = 'a'
    mmkt, wmkt = tsc_market(df, code)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(wmkt, label='a wght', color='blue')
    for tpl in mmkt:
        plt.plot(tpl[0], tpl[1], label=tpl[2])
    plt.legend()
    fig.savefig('%s.jpg' % code)
    plt.show()
