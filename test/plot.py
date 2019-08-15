# -*- coding:utf-8 -*-
import tushare as ts
import pandas as pd
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 6))
    k = 8
    x = list(range(4 + k, 100, 2))
    y1, y2 = [], []
    for n in x:
        t1 = ((n-1)/2 + 1 + 0.5) / n**2
        y1.append(t1)
        mn_t2 = 1
        for k in (2, n-4, 2):
            t2 = (n-k)*(n-k-1)
            if t2%(2*n) == 0:
                t2 = t2/2/n
            else:
                t2 = t2/2/n + 1
            t2 = t2/(n-k)**2
            if t2 < mn_t2:
                mn_t2 = t2
        y2.append(mn_t2)
    plt.plot(x, y1, label='y1', color='blue')
    plt.plot(x, y2, label='y2', color='red')
    plt.legend()
    plt.show()
