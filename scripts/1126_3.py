#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math

def queue(src, distance, time):
    data = np.append(src[1:], np.array([[distance,time]]),axis=0)
    return data


def console_print(data):
    ave = np.average(data[:,0]) # 平均の計算
    sigma = np.std(data[:,0])   # 標準偏差の計算
    sn = 20 * np.log10(ave/sigma) # SN比の計算

    # 時刻・距離・S/N比の表示
    print('--------------------------------')
    print("時刻：", datetime.datetime.today())
    print("距離：" + str(data[-1,0]) + "[cm]")
    print("S/N比：" + str(sn) + "[dB]")
    #print(np.transpose(data))

#def plot_distance(data):


def main():
    # カウント用変数
    #i = 0

    # 1次元配列の生成(距離データ格納用)
    data = np.zeros((40,2),dtype=np.float32)

    # start_time
    start_time = time.time()

    # シリアル接続するポート
    ser = serial.Serial("/dev/ttyUSB0", 9600)

    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    #ax_2 = fig.add_subplot(212)

    try:
        while(True):
            # シリアルデータを受信して距離データを取得
            line = ser.readline()    # 行終端まで読み込む
            line = line.decode()     # byteからstringに変換
            #print(line)
            # byteからstringに変換
            distance = line.rstrip() # 行終端コード削除
            print(distance)
            # キュー操作
            data = queue(data, float(distance), time.time())

            # コンソールに結果表示
            console_print(data)

            # 開始後何秒経過したか
            print("time = ",time.time() - start_time)

            #line, = ax.plot(data, color='red')
            time_axis = data[:,1] - time.time()
            ax_1.clear()
            ax_1.plot(time_axis, data[:,0], color="k")
            ax_1.set_xlabel("time[s]")
            ax_1.set_ylabel("distance[cm]")
            ax_1.set_ylim([-10,220])

#            ax_2.clear()

#            ax_2.plot(time_axis, np.sin(time_axis), color="k")
            # ax_2.set_xlabel("time[s]")
            # ax_2.set_ylabel("velocity[cm]")
            # ax_2.set_ylim([-10,220])

            plt.pause(0.01)

            # #line.remove()
            # #print("line = ", line)
            # #print("distance =",distance)
            # #i+=1

    except KeyboardInterrupt:
        ser.close()
        print("End")


if __name__ == '__main__':
    main()
