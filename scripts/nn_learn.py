#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import nn

def main():
    epoch, step_diff, lay1, lay2 = nn.nn_3()

    plt.plot(epoch, step_diff,label="3_layer")
    plt.xlabel("epoch time")
    plt.ylabel("step_diff")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
