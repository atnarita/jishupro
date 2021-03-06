#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# モジュールのimport
import numpy as np
import matplotlib.pyplot as plt
import serial
import datetime
import time
import struct
import math
import nn
import threading
import cv2

# 単位は[m],[s]

# グローバル変数の宣言
time_list = np.zeros((30),dtype=np.float32) # 30フレームの時間を保存（UNIX時間）
start_time = 0.0 # スタートしたときのUNIX時間

acc_src = np.zeros((3),dtype=np.float32) # arduinoから送られてきた加速度情報
acc_array = np.zeros((30,3),dtype=np.float32) # 30フレーム(x,y,z)の情報を保存
acc_list = np.zeros((3),dtype=np.float32) # [m/s^2]似直したときの加速度
acc_array_modi = np.zeros((30,3),dtype=np.float32) # 30フレーム(x,y,z)の処理した後の情報を保存
acc_list_modi = np.zeros((3),dtype=np.float32) # [m/s^2]似直したときの処理した後の加速度を保存

# 初期姿勢での加速度
x_acc_zero = 0
y_acc_zero = 0
z_acc_zero = 0

speed_array = np.zeros((30,3),dtype=np.float32) # 30フレーム分の速度情報[m/s]
speed_list = np.zeros((3),dtype=np.float32) # 今の速度[m/s]

distance_array = np.zeros((30,3),dtype=np.float32) # 30フレーム分の距離情報[m]
distance_list = np.zeros((3),dtype=np.float32) # 1フレームでの移動距離[m]

step_time = 0 # 歩数
obstacle = 500 # 障害物との距離

lay1 = nn.start_layer(90,60)
lay2 = nn.last_layer(60,1)
lay1.weight = np.array(np.loadtxt('data/w1.txt', dtype='float32'))
lay2.weight = np.array(np.loadtxt('data/w2.txt', dtype='float32'))

count = 1 # 何フレーム目なのかを表す
step_num = 0 # 歩数
step_flag = 0 # 歩行状態

#ser = serial.Serial("/dev/ttyUSB0",9600) # こっちになることもある
ser = serial.Serial("/dev/ttyUSB1",9600) # serial通信


def acc_func():
    """
    生データの加速度値を受け取って[m/s^2]に変換し、各配列を更新する。
    """
    global acc_src
    global acc_array
    global acc_list
    global acc_array_modi
    global acc_list_modi
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero

    acc_list = np.array([(acc_src[0]-x_acc_zero)/(220/9.8), (acc_src[1]-y_acc_zero)/(220/9.8), (acc_src[2]-z_acc_zero)/(220/9.8)])
    acc_array = np.append(acc_array[1:], [acc_list], axis=0)

    if np.std(acc_array[-5:,:]) < 0.5:
        acc_list_modi = np.zeros(3)
    else:
        acc_list_modi = acc_list
    acc_array_modi = np.append(acc_array_modi[1:], [acc_list_modi], axis=0)


def speed_func():
    """
    加速度を積分して(x,y,z)方向それぞれの速度を計算する。その後、各配列を更新する。
    """
    global time_list
    global speed_array
    global speed_list
    global acc_list

    time_diff = time_list[-1] - time_list[-2]
    speed_list = speed_list + acc_list * time_diff
    speed_array = np.append(speed_array[1:], [speed_list], axis=0)


def distance_func():
    """
    速度を積分して(x,y,z)方向の移動距離をそれぞれ計算する。その後、各配列を更新する。
    """
    global time_list
    global distance_array
    global distance_list
    global speed_list

    time_diff = time_list[-1] - time_list[-2]
    distance_list = distance_list + speed_list * time_diff
    distance_array = np.append(distance_array[1:], [distance_list], axis=0)


def time_func():
    """
    各フレームの時間を格納する配列を更新する。
    """
    global start_time
    global time_list

    time_list = np.append(time_list[1:], time.time())


def step():
    """
    歩数を計測する
    """
    global step_num
    global step_flag

    # 標準偏差の計算
    sigma_x = np.std(acc_array[-5:,0])
    sigma_y = np.std(acc_array[-5:,1])
    sigma_z = np.std(acc_array[-5:,2])
    acc_ave = np.average(acc_array[-3:,0])

    if (math.sqrt(sigma_z**2 + sigma_y**2 + sigma_z**2) >= 0.5) and (acc_ave > 0) and (acc_ave > 1) and step_flag == 0:
        step_num += 1
        step_flag = 1
    if acc_array[-1,0] < 0:
        step_flag = 0


def update():
    """
    シリアル通信でArduinoからの情報（x,y,z,障害物との距離）を受け取り、各変数を更新する。
    ターミナルに情報をprintする。
    main関数からのマルチスレッドで実行
    """
    global ser
    global time_list
    global start_time
    global acc_src
    global acc_array
    global acc_list
    global acc_list_modi
    global x_acc_zero
    global y_acc_zero
    global z_acc_zero
    global speed_arra
    global speed_lis英語t
    global distance_array
    global distance_list
    global step_time
    global count_loop
    global ser
    global obstacle
    global count

    path = 0 # NNを用いたときの移動距離

    print("thread(update) : start")
    start_time = time.time()

    while(True):
        # シリアルデータを受信して距離データを取得
        line = ser.readline()    # 行終端まで読み込む
        line = line.decode()     # byteからstringに変換
        ser_data = list(map(int, line.rstrip().split()))

        acc_src = np.array([ser_data[0], ser_data[1], ser_data[2]]) #acc_srcに変数を入れる
        obstacle = ser_data[3] #obshacaleに変数を入れる

        # 更新
        time_func()
        acc_func()
        speed_func()
        distance_func()
        step()

        # ターミナルに結果表示
        print("time = ",time.time() - start_time) # 開始後何秒経過したか
        print("acc_list= {}".format(acc_list)) # 移動加速度を出す
        print("acc_list_modi= {}".format(acc_list_modi)) # 移動加速度を出す
        print("speed_list = {}".format(speed_list)) # 移動速度を出す
        print("distance_list = {}".format(distance_list)) # 積分から得た移動速度を出す
        print("step = {}".format(step_num))

        # 30フレーム（約3病）に一回はNNで距離を新しくする
        if count%30==0:
            print("\007",end="")
            output = lay1.forward(acc_array.reshape(90))
            output = lay2.forward(output)
            path += float(output)

        print("path(NN) = {}\n".format(path))

        count += 1
        time.sleep(0.1) # 0.1[s]待つ

def main():
    """
    ・初期姿勢のキャリブレーション
    ・カメラ画像の表示
    ・センサ情報のプロットを行う
    """
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
    global obstacle


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

    try:
        # マルチプロセスだと情報のやり取りのせいなのか上手く行かない
        th1 = threading.Thread(target=update, args=())
        th1.start() # update関数を別スレッドで開始

        # 各種データのプロットについて
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)

        time_axis = time_list[:] - time.time()
        lines_x, = ax_1.plot(time_axis, acc_array[:,0], color="r",label="x")
        lines_y, = ax_1.plot(time_axis, acc_array[:,1], color="g",label='y')
        lines_z, = ax_1.plot(time_axis, acc_array[:,2], color="b",label='z')

        ax_1.set_xlabel("time[s]")
        ax_1.set_ylabel("[m/s^2]")
        ax_1.set_ylim([-10,10])
        ax_1.set_xlim([-3.0, 0.0])
        ax_1.legend()

        # カメラ画像について
        capture = cv2.VideoCapture(1)
        cv2.namedWindow("Capture")

        while(True):
            # カメラ画像をwindowに表示
            ret, frame = capture.read()

            # 40 <  (障害物との距離)       ... 何もしない
            # 15 <= (障害物との距離) <= 40 ... 警告
            #       (障害物との距離) <  15 ... 危険を知らせる
            if 15 <= obstacle and obstacle <= 40 :
                text = str(obstacle) + "[cm]. Take Care!"
                cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 5, 8)
            elif obstacle < 15:
                text = str(obstacle) + "[cm]. Dangerous!"
                cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 5, 8)
            else:
                pass

            cv2.imshow("Capture", frame)
            c = cv2.waitKey(2)
            if c == 27:
                capture.release()
                cv2.destroyAllWindows()
                break


            # プロットの準備とプロット
            time_axis = time_list[:] - time.time() # 時間軸の作成

            # リアルタイムのプロットをするためにグラフ部分だけを更新
            lines_x.set_data(time_axis, acc_array[:,0])
            lines_y.set_data(time_axis, acc_array[:,1])
            lines_z.set_data(time_axis, acc_array[:,2])

            # wait for 0.01[s]
            plt.pause(0.01)

    except KeyboardInterrupt:
        ser.close()
        capture.release()
        cv2.destroyAllWindows()
        pr1.join()
        print("\nThank you...")


if __name__ == '__main__'    :
    main()
