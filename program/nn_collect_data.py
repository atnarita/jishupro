#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math

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

    time_list = np.append(time_list[1:], time.time()-start_time)



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

    #シリアル通信
    ser = serial.Serial("/dev/ttyUSB1",9600)
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
            if len(ser_data) != 4:
                print("sorry, one more time...")
            else:
                x_acc_zero = ser_data[0]
                y_acc_zero = ser_data[1]
                z_acc_zero = ser_data[2]
                print("let's start")
                break

    # fig = plt.figure()
    # ax_1 = fig.add_subplot(111)

    file_acc = "data/training_acc.txt"
    file_dis = "data/training_dis.txt"
    fileobj_acc = open(file_acc, "w", encoding = "utf_8")
    fileobj_dis = open(file_dis, "w", encoding = "utf_8")

    start_time = time.time()
    count = 1

    try:
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
            # console_print(x_acc, y_acc, z_acc)
            # 開始後何秒経過したか
            print("time = ",time.time() - start_time)
            # 移動速度を出す
            # print("acc_list={}".format(acc_list))
            # print("speed_list={}".format(speed_list))
            # print("distance_list={}".format(distance_list))

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

            fileobj_acc.write(str(acc_list[0])+" "+str(acc_list[1])+" "+str(acc_list[2])+" ")
            if count%30==0:
                print("\007")
                ans_dis = input("how far did you move ? : ")
                fileobj_acc.write("\n")
                fileobj_dis.write(ans_dis)
                fileobj_dis.write("\n")
                fin = input("continue ? (y or n): ")
                if fin == "n":
                    break



            count += 1
            time.sleep(0.1)


    except KeyboardInterrupt:
        ser.close()
        print("Thank you...")

    fileobj_acc.close()
    fileobj_dis.close()


if __name__ == '__main__'    :
    main()
