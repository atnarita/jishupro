#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math

N = 20
x_acc = np.zeros((N),dtype=np.float32)
y_acc = np.zeros((N),dtype=np.float32)
z_acc = np.zeros((N),dtype=np.float32)
time_list = np.zeros((N),dtype=np.float32)

x_acc_zero = 0
y_acc_zero = 0
z_acc_zero = 0

dist = 0.0
step_time = 0


def acc_queue_for_x(src, acc):
    global x_acc_zero
    acc = (acc - x_acc_zero)/(220/9.8)
    data = np.append(src[1:], acc)
    return data

def acc_queue_for_y(src, acc):
    global y_acc_zero
    acc = (acc - y_acc_zero)/(220/9.8)
    data = np.append(src[1:], acc)
    return data

def acc_queue_for_z(src, acc):
    global z_acc_zero
    acc = (acc - z_acc_zero)/(220/9.8)
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

def distance(src, time):
    global dist
    sigma_x = np.std(x_acc[-5:])   # 標準偏差の計算
    sigma_y = np.std(y_acc[-5:])
    sigma_z = np.std(z_acc[-5:])

    acc_ave = np.average(src[-5:])
    delta_t = time[-1] - time[-5]
    if sigma_z >= 0.5:
        dist += acc_ave * (delta_t**2)

def step():
    global step_time
    sigma_x = np.std(x_acc[-5:])   # 標準偏差の計算
    sigma_y = np.std(y_acc[-5:])
    sigma_z = np.std(z_acc[-5:])
    if math.sqrt(sigma_z**2 + sigma_y**2 + sigma_z**2) >= 10.0:
        step_time += 1



def console_print(x_acc,y_acc,z_acc):
    ave_x = np.average(x_acc) # 平均の計算
    ave_y = np.average(y_acc)
    ave_z = np.average(z_acc)
    sigma_x = np.std(x_acc[:])   # 標準偏差の計算
    sigma_y = np.std(y_acc[:])
    sigma_z = np.std(z_acc[:])
    # sn_x = 20 * np.log10(ave_x/sigma_x) # SN比の計算
    # sn_y = 20 * np.log10(ave_y/sigma_y)
    # sn_z = 20 * np.log10(ave_z/sigma_z)

    # 時刻・距離・S/N比の表示
    print('--------------------------------')
    print("時刻：", datetime.datetime.today())

    print("X : 加速度(a_x):{:6f}\t分散(sigma_x):{:6f}\t速度(v_x): {:6f}".format(x_acc[-1], sigma_x, speed(x_acc, time_list)))
    print("Y : 加速度(a_y):{:6f}\t分散(sigma_y):{:6f}\t速度(v_y): {:6f}".format(y_acc[-1], sigma_y, speed(y_acc, time_list)))
    print("Z : 加速度(a_z):{:6f}\t分散(sigma_z):{:6f}\t速度(v_z): {:6f}".format(z_acc[-1], sigma_z, speed(z_acc, time_list)))

    distance(z_acc, time_list)
    step()
    print("総移動距離 : {:6f}".format(dist))
    print("歩数 : {}".format(step_time))

    # print("S/N比\tx:" + str(sn_x) + "\ty:" + str(sn_y) + "\tz:" + str(sn_z) + "[dB]")
    #print(np.transpose(data))

def fft(src,time_list):
    F = np.fft.fft(src)
    Amp = np.abs(F)
    freq = np.linspace(0, time_list[-1] - time_list[0], N)

    # 正規化 + 交流成分2倍
    F = F/(N/2)
    F[0] = F[0]/2

    return freq, Amp



def main():
    #グローバル変数を関数の中で使う
    global x_acc
    global y_acc
    global z_acc
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero
    global time_list

    #シリアル通信
    ser = serial.Serial("/dev/ttyUSB0",9600)
    #ser = serial.Serial("/dev/ttyS4", 9600)
    print("connected")

    #キャリブレーションを行う
    while True:
        print("press 's'(start) or 'q'(quit) : ",end="")
        char = input()
        if char == "q":
            print("finished")
            return
        elif char == "s":
            line = ser.readline()    # 行終端まで読み込む
            line = line.decode()     # byteからstringに変換
            ser_data = list(map(int, line.rstrip().split()))
            if len(ser_data) != 3:
                print("sorry, one more time...")
            else:
                x_acc_zero = ser_data[0]
                y_acc_zero = ser_data[1]
                z_acc_zero = ser_data[2]
                print("let's start")
                break

    fig = plt.figure()
    ax_1 = fig.add_subplot(111)

    start_time = time.time()
    try:
        while(True):
            # シリアルデータを受信して距離データを取得
            line = ser.readline()    # 行終端まで読み込む
            line = line.decode()     # byteからstringに変換
            ser_data = list(map(int, line.rstrip().split()))
            print(ser_data)

            #それぞれキュー操作
            x_acc = acc_queue_for_x(x_acc, ser_data[0])
            y_acc = acc_queue_for_y(y_acc, ser_data[1])
            z_acc = acc_queue_for_z(z_acc, ser_data[2])
            time_list = time_queue(time_list, time.time())

            # ターミナルに結果表示
            console_print(x_acc, y_acc, z_acc)
            # 開始後何秒経過したか
            print("time = ",time.time() - start_time)
            # 移動速度を出す

            freq_z_acc, amp_z_acc = fft(z_acc, time_list)

            # プロットの準備とプロット
            # time_axis = time_list[:] - time.time()
            # # fo_x_acc = np.fft.fft(x_acc)
            # # fo_y_acc = np.fft.fft(y_acc)
            # # fo_z_acc = np.fft.fft(z_acc)
            # # fo_z_acc = fo_z_acc/(N/2)
            ax_1.clear()
            # ax_1.plot(time_axis, x_acc, color="r",label="x")
            # ax_1.plot(time_axis, y_acc, color="g",label='y')
            ax_1.plot(freq_z_acc, amp_z_acc, color="b",label='z')
            ax_1.set_xlabel("Frequency")
            ax_1.set_ylabel("Amplitude")
            ax_1.legend()
            #ax_1.set_ylim([-10,10])

            plt.pause(0.01)


    except KeyboardInterrupt:
        ser.close()
        print("Thank you...")


if __name__ == '__main__'    :
    main()
