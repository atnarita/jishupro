#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math
import nn
import threading

# 単位は[m],[s]
time_list = np.zeros((30),dtype=np.float32)
start_time = 0.0

acc_src = np.zeros((3),dtype=np.float32)
acc_array = np.zeros((30,3),dtype=np.float32)
acc_list = np.zeros((3),dtype=np.float32)
x_acc_zero = 0
y_acc_zero = 0
z_acc_zero = 0

speed_array = np.zeros((30,3),dtype=np.float32)
speed_list = np.zeros((3),dtype=np.float32)

distance_array = np.zeros((30,3),dtype=np.float32)
distance_list = np.zeros((3),dtype=np.float32)

step_time = 0


lay1 = nn.start_layer(90,60)
lay2 = nn.last_layer(60,1)
lay1.weight = np.array(np.loadtxt('data/w1.txt', dtype='float32'))
lay2.weight = np.array(np.loadtxt('data/w2.txt', dtype='float32'))

start_time = 0
count = 1
ser = serial.Serial("/dev/ttyUSB0",9600)


def acc_func():
    global acc_src
    global acc_array
    global acc_list
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero

    acc_list = np.array([(acc_src[0]-x_acc_zero)/(220/9.8), (acc_src[1]-y_acc_zero)/(220/9.8), (acc_src[2]-z_acc_zero)/(220/9.8)])
    acc_array = np.append(acc_array[1:], [acc_list], axis=0)


def speed_func():
    global time_list
    global speed_array
    global speed_list
    global acc_list

    time_diff = time_list[-1] - time_list[-2]
    speed_list = speed_list + acc_list * time_diff
    speed_array = np.append(speed_array[1:], [speed_list], axis=0)
    # if np.std(acc_array[-5:,:]) < 0.5:
    #     acc_list = np.zeros(3)
    #     speed_list = np.zeros(3)


def distance_func():
    global time_list
    global distance_array
    global distance_list
    global speed_list

    time_diff = time_list[-1] - time_list[-2]
    distance_list = distance_list + speed_list * time_diff
    distance_array = np.append(distance_array[1:], [distance_list], axis=0)


def time_func():
    global start_time
    global time_list

    time_list = np.append(time_list[1:], time.time())

def drawing():
    global time_list
    global start_time
    global acc_src
    global acc_array
    global acc_list
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero
    global speed_arra
    global speed_list
    global distance_array
    global distance_list
    global step_time
    global count_loop
    global ser

    print("thread(drawing) : start")

    fig = plt.figure()
    ax_1 = fig.add_subplot(111)

    while(True):
        # プロットの準備とプロット
        time_axis = time_list[:] - time.time()
        print("Hi!")
        ax_1.clear()
        ax_1.plot(time_axis, acc_array[:,0], color="r",label="x")
        ax_1.plot(time_axis, acc_array[:,1], color="g",label='y')
        ax_1.plot(time_axis, acc_array[:,2], color="b",label='z')
        ax_1.set_xlabel("time[s]")
        ax_1.set_ylabel("[m/s^2]")
        ax_1.legend()
        ax_1.set_ylim([-10,10])

        plt.pause(0.01)



def update():
    global ser
    global time_list
    global start_time
    global acc_src
    global acc_array
    global acc_list
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero
    global speed_arra
    global speed_list
    global distance_array
    global distance_list
    global step_time
    global count_loop
    global ser
    path = 0
    count = 0
    print("thread(update) : start")
    start_time = time.time()

    while(True):
        # シリアルデータを受信して距離データを取得
        line = ser.readline()    # 行終端まで読み込む
        line = line.decode()     # byteからstringに変換
        ser_data = list(map(int, line.rstrip().split()))
        # print(ser_data)

        #acc_srcに変数を入れる
        acc_src = np.array([ser_data[0], ser_data[1], ser_data[2]])
        time_func()
        acc_func()
        speed_func()
        distance_func()

        # ターミナルに結果表示
        # 開始後何秒経過したか
        print("time = ",time.time() - start_time)
        # 移動速度を出す
        print("acc_list={}".format(acc_list))
        print("speed_list={}".format(speed_list))
        print("distance_list={}".format(distance_list))

        # # プロットの準備とプロット
        # time_axis = time_list[:] - time.time()
        # ax_1.clear()
        # ax_1.plot(time_axis, x_acc[:], color="r",label="x")
        # ax_1.plot(time_axis, y_acc[:], color="g",label='y')
        # ax_1.plot(time_axis, z_acc[:], color="b",label='z')
        # ax_1.set_xlabel("time[s]")
        # ax_1.set_ylabel("[m/s^2]")
        # ax_1.legend()
        # ax_1.set_ylim([-10,10])
        #
        # plt.pause(0.01)

        if count%30==0:
            print("\007",end="")
            output = lay1.forward(acc_array.reshape(90))
            output = lay2.forward(output)
            path += float(output)
        print("path(NN) = {}\n".format(path))

        count += 1
        time.sleep(0.1)

def main():
    #グローバル変数を関数の中で使う
    global time_list
    global start_time
    global acc_src
    global acc_array
    global acc_list
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero
    global speed_arra
    global speed_list
    global distance_array
    global distance_list
    global step_time
    global count_loop
    global ser

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

    try:
        # th1 = threading.Thread(target=update)
        # th2 = threading.Thread(target=drawing)
        # th1.start()
        # th2.start()

        pr1 = multiprocessing.Process(target=update, args=())
        # pr2 = multiprocessing.Process(target=drawing, args=())
        # pr2.start()
        pr1.start()

    except KeyboardInterrupt:
        ser.close()
        pr1.join()
        pr2.join()
        print("\nThank you...")




if __name__ == '__main__'    :
    main()
