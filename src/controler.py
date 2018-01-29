# -*- coding: utf-8 -*-
"""
# @Time    : 2018/1/28 下午11:09
# @Author  : zhanzecheng
# @File    : controler.py
# @Software: PyCharm
"""
import math
import os
import shutil
from ios_env import IOS
from BL_brain import QLearningTable
from PIL import ImageDraw, Image
import time
import random

class Controler:
    
    def __init__(self):
        self.Under_game_score_y = 200
        self.Piece_base_height_1_2 = 13
        self.Piece_body_width = 49
        self.Screenshot_backup_dir = 'backups/'
        self.x = 0
        self.y = 0
        self.environment = IOS()
        self.brain = QLearningTable(learning=True)

    def build_state(self):
        # the distance between start point and target point
        self.environment.pull_screenshot()
        im = Image.open("./1.png")
        # 使用map函数来对每一个对都用int
        piece_x, piece_y, board_x, board_y = map(int, self.find_piece_and_board(im))
        self.x = board_x
        self.y = board_y
        print('location: ', piece_x, piece_y, board_x, board_y)
        state = self.brain.build_state(piece_x, piece_y, board_x, board_y)
        return state

    def jump(self):
        print("## Start ...")
        state = self.build_state()
        action = self.brain.choose_action(state)
        # tap hold
        # time.sleep(random.uniform(1, 1.1))
        self.environment.tap_hold(action, 85, 500, )
        time.sleep(2)
        isOver, bias = self.is_over()
        print("isOver: ", isOver)
        if state == 0:
            bias = 1
        if not isOver:
            self.brain.update(action, state, bias)
        else:
            bias = 5
            self.brain.update(action, state, bias)
            time.sleep(random.uniform(1, 1.1))
        self.brain.add_t()
        print("state:" + str(state) + ",action:" + str(action) + ",reward:" + str(
            float("{0:.2f}".format(bias * self.brain.penalty + 0.2))))
        # print('self.Q')
        # for k,v in self.Q.items():
        #    print(k,v)
        return isOver


    def is_over(self):
        is_over = False
        # check if the game is over
        time.sleep(random.uniform(0.7, 1.1))
        # print("snapchat!!!")
        self.environment.pull_screenshot()
        im = Image.open("./1.png")
        ts = int(time.time())
        piece_x, piece_y, board_x, board_y = map(int, self.find_piece_and_board(im))
        self.save_debug_creenshot(ts, im, piece_x, piece_y, board_x, board_y)
        self.backup_screenshot(ts)
        bias = int(math.sqrt(math.fabs(board_x - self.x) ** 2 + math.fabs(board_y - self.y) ** 2) / 100)
        print("bias: ", bias)
        # print("last x: ",self.x)
        # print("last y: ", self.y)
        if (piece_x == 0 and piece_y == 0) or bias > 10:
            is_over = True
        self.x = piece_x
        self.y = piece_y
        '''
        print("new x:",piece_x)
        print("new y:",piece_y)
        print("new target x:", board_x)
        print("new target y:" ,board_y)'''
        return is_over, bias

    def backup_screenshot(self,ts):
        if not os.path.isdir(self.Screenshot_backup_dir):
            os.mkdir(self.Screenshot_backup_dir)
        shutil.copy('1.png', '{}{}.png'.format(self.Screenshot_backup_dir, ts))

    def save_debug_creenshot(self,ts, im, piece_x, piece_y, board_x, board_y):
        draw = ImageDraw.Draw(im)
        draw.line((piece_x, piece_y) + (board_x, board_y), fill=2, width=3)
        draw.line((piece_x, 0, piece_x, im.size[1]), fill=(255, 0, 0))
        draw.line((0, piece_y, im.size[0], piece_y), fill=(255, 0, 0))
        draw.line((board_x, 0, board_x, im.size[1]), fill=(0, 0, 255))
        draw.line((0, board_y, im.size[0], board_y), fill=(0, 0, 255))
        draw.ellipse((piece_x - 10, piece_y - 10, piece_x + 10, piece_y + 10), fill=(255, 0, 0))
        draw.ellipse((board_x - 10, board_y - 10, board_x + 10, board_y + 10), fill=(0, 0, 255))
        del draw
        im.save('{}{}_d.png'.format(self.Screenshot_backup_dir, ts))

    def find_piece_and_board(self, im):
        w, h = im.size

        print("size: {}, {}".format(w, h))

        piece_x_sum = piece_x_c = piece_y_max = 0
        board_x = board_y = 0
        scan_x_border = int(w / 8)  # 扫描棋子时的左右边界
        scan_start_y = 0  # 扫描的起始 y 坐标
        im_pixel = im.load()

        # 以 50px 步长，尝试探测 scan_start_y
        for i in range(self.Under_game_score_y, h, 50):
            last_pixel = im_pixel[0, i]
            for j in range(1, w):
                pixel = im_pixel[j, i]

                # 不是纯色的线，则记录scan_start_y的值，准备跳出循环
                if pixel != last_pixel:
                    scan_start_y = i - 50
                    break

            if scan_start_y:
                break

        print("scan_start_y: ", scan_start_y)

        # 从 scan_start_y 开始往下扫描，棋子应位于屏幕上半部分，这里暂定不超过 2/3
        for i in range(scan_start_y, int(h * 2 / 3)):
            # 横坐标方面也减少了一部分扫描开销
            for j in range(scan_x_border, w - scan_x_border):
                pixel = im_pixel[j, i]
                # 根据棋子的最低行的颜色判断，找最后一行那些点的平均值，这个颜
                # 色这样应该 OK，暂时不提出来
                if (50 < pixel[0] < 60) \
                        and (53 < pixel[1] < 63) \
                        and (95 < pixel[2] < 110):
                    piece_x_sum += j
                    piece_x_c += 1
                    piece_y_max = max(i, piece_y_max)

        if not all((piece_x_sum, piece_x_c)):
            return 0, 0, 0, 0
        piece_x = piece_x_sum / piece_x_c
        piece_y = piece_y_max - self.Piece_base_height_1_2  # 上移棋子底盘高度的一半

        for i in range(int(h / 3), int(h * 2 / 3)):
            last_pixel = im_pixel[0, i]
            if board_x or board_y:
                break
            board_x_sum = 0
            board_x_c = 0

            for j in range(w):
                pixel = im_pixel[j, i]
                # 修掉脑袋比下一个小格子还高的情况的 bug
                if abs(j - piece_x) < self.Piece_body_width:
                    continue

                # 修掉圆顶的时候一条线导致的小 bug，这个颜色判断应该 OK，暂时不提出来
                if abs(pixel[0] - last_pixel[0]) \
                        + abs(pixel[1] - last_pixel[1]) \
                        + abs(pixel[2] - last_pixel[2]) > 10:
                    board_x_sum += j
                    board_x_c += 1

            if board_x_sum:
                board_x = board_x_sum / board_x_c

        # 按实际的角度来算，找到接近下一个 board 中心的坐标 这里的角度应该
        # 是 30°,值应该是 tan 30°, math.sqrt(3) / 3
        board_y = piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3

        if not all((board_x, board_y)):
            return 0, 0, 0, 0

        return piece_x, piece_y, board_x, board_y


if __name__ == '__main__':
    # Create agent
    agent = Controler()
    n_test = 5
    n_train = 2000
    tolerance = 0.005
    log_metrics = True
    if agent.brain.learning:
        log_filename = os.path.join("logs", "sim_improved-learning.csv")
    else:
        log_filename = os.path.join("logs", "sim_no-learning.csv")
    table_filename = os.path.join("logs", "sim_improved-learning.txt")
    table_file = open(table_filename, 'w')
    log_file = open(log_filename, 'w')
    total_trials = 1
    testing = False
    trial = 1
    # Run
    while True:
        print("trial:", trial)
        print("total_trials:", total_trials)

        if testing:
            if trial > n_test:
                break
        else:
            if trial > n_train:
                testing = True
                trial = 1

        # Pretty print to terminal
        print()
        print("/-------------------------")
        if testing:
            print("| Testing trial {}".format(trial))
        else:
            print("| Training trial {}".format(trial))

        print("\-------------------------")
        print()
        # Increment
        trial += 1
        total_trials = total_trials + 1

        current_time = 0.0
        last_updated = 0.0
        start_time = time.time()
        # current_time = time.time() - start_time
        isOver = False
        while not isOver:
            isOver = agent.jump()
            agent.brain.reset(testing)
            if isOver:
                agent.environment.restart()
        agent.brain.save_Q()
        # Clean up
        if log_metrics:
            if agent.brain.learning:
                f = table_file

                f.write("/-----------------------------------------\n")
                f.write("| State-action rewards from Q-Learning\n")
                f.write("\-----------------------------------------\n\n")

                for state in agent.brain.Q:
                    f.write("{}\n".format(state))
                    for action, reward in agent.brain.Q[state].items():
                        f.write(" -- {} : {:.2f}\n".format(action, reward))
                    f.write("\n")
        print("\nSimulation ended. . . ")
    table_file.close()
    log_file.close()