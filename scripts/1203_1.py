#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math

def acc_queue_for_x(src, acc):
    acc = (acc - 507)/(230/9.8)
    data = np.append(src[1:], acc)
    return data

def acc_queue_for_y(src, acc):
    acc = (acc - 502.5)/(223/9.8)
    data = np.append(src[1:], acc)
    return data

def acc_queue_for_z(src, acc):
    acc = (acc - 505)/(204/9.8)
    data = np.append(src[1:], acc)
    return data

def time_queue(src, time):
    data = np.append(src[1:], time)
    return data

def speed(src, time):
    acc_ave = np.average(src[-3:])
    delta_t = time[-1] - time[-3]
    km_per_h = ((acc_ave * delta_t) * 60 * 60)/1000
    return km_per_h

def console_print(x_acc,y_acc,z_acc):
    ave_x = np.average(x_acc) # 平均の計算
    ave_y = np.average(y_acc)
    ave_z = np.average(z_acc)
    # sigma_x = np.std(x_acc[:])   # 標準偏差の計算
    # sigma_y = np.std(y_acc[:])
    # sigma_z = np.std(z_acc[:])
    # sn_x = 20 * np.log10(ave_x/sigma_x) # SN比の計算
    # sn_y = 20 * np.log10(ave_y/sigma_y)
    # sn_z = 20 * np.log10(ave_z/sigma_z)

    # 時刻・距離・S/N比の表示
    print('--------------------------------')
    print("時刻：", datetime.datetime.today())
    print("加速度\tx: {:6f}\ty: {:6f}\tz: {:6f}".format(x_acc[-1],y_acc[-1],z_acc[-1]))
    # print("S/N比\tx:" + str(sn_x) + "\ty:" + str(sn_y) + "\tz:" + str(sn_z) + "[dB]")
    #print(np.transpose(data))

#def plot_distance(data):


def main():
    # 1次元配列の生成(距離データ格納用)
    # while True:
    #     char = input()
    #     if char == "q":
    #         return
    #     elif char == "s":
    #         ser = serial.Serial("/dev/ttyUSB0", 9600)
    #         line = ser.readline()    # 行終端まで読み込む
    #         line = line.decode()     # byteからstringに変換
    #         ser_data = list(map(int, line.rstrip().split()))
    #         if len(ser_data) != 3:
    #             pass
    #         else:
    #             x_acc_zero = ser_data[0]
    #             y_acc_zero = ser_data[1]
    #             z_acc_zero = ser_data[2]
    #             break
    #     else:
    #         pass


    x_acc = np.zeros((20),dtype=np.float32)
    y_acc = np.zeros((20),dtype=np.float32)
    z_acc = np.zeros((20),dtype=np.float32)
    time_list = np.zeros((20),dtype=np.float32)

    # start_time
    start_time = time.time()

    # シリアル接続するポート
    ser = serial.Serial("/dev/ttyUSB0", 9600)
    #ser = serial.Serial("/dev/ttyS4", 9600)
    print("connected!")
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)

    try:
        while(True):
            # シリアルデータを受信して距離データを取得
            line = ser.readline()    # 行終端まで読み込む
            line = line.decode()     # byteからstringに変換
            ser_data = list(map(int, line.rstrip().split()))
            print(ser_data)

            # byteからstringに変換
            #print(ser_data)
            # キュー操作
            x_acc = acc_queue_for_x(x_acc, ser_data[0])
            y_acc = acc_queue_for_y(y_acc, ser_data[1])
            z_acc = acc_queue_for_z(z_acc, ser_data[2])
            time_list = time_queue(time_list, time.time())

            #print(x_acc)

             # コンソールに結果表示
            console_print(x_acc, y_acc, z_acc)

            # 開始後何秒経過したか
            print("time = ",time.time() - start_time)
            print(speed(z_acc,time_list))

            #line, = ax.plot(data, color='red')
            time_axis = time_list[:] - time.time()
            ax_1.clear()
            ax_1.plot(time_axis, x_acc[:], color="k",label="x")
            ax_1.plot(time_axis, y_acc[:], color="b",label='y')
            ax_1.plot(time_axis, z_acc[:], color="r",label='z')
            ax_1.set_xlabel("time[s]")
            ax_1.set_ylabel("[m/s^2]")
            ax_1.legend()
            ax_1.set_ylim([-25,25])

            plt.pause(0.01)

    except KeyboardInterrupt:
        ser.close()
        print("End")


if __name__ == '__main__':
    main()
