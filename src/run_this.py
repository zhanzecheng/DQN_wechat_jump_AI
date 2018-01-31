# -*- coding: utf-8 -*-
"""
# @Time    : 2018/1/31 下午5:35
# @Author  : zhanzecheng
# @File    : run_this.py
# @Software: PyCharm
"""

import tensorflow as tf
import numpy as np
import random
import tqdm
from RL_brain import DuelingDQN
from Env import Env

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_bool("pre", True, "use the pre_train model")
tf.app.flags.DEFINE_integer("pre_epoch", 1000000, "the epoch of pre_train model")

'''
为了截图顺利我设置的延时时间有一点长（3秒），大家可以根据自己的需要修改Env.py文件
'''
def pre_train(pre_epoch):
    '''
    由于真实环境游戏太慢样本太少，这里使用模拟游戏环境的方法来进行模型的预训练。
    '''
    #初始化第一个状态值
    print('====================开始预训练模型====================')
    state = env.generate_reset_state()
    for i in tqdm.tqdm(range(pre_epoch)):
        action = RL.choose_action(state)
        # 换算成真正的时间ms
        action_ = (action) * 50 + 300
        # 利用 https://github.com/wangshub/wechat_jump_game 提供的按压系数来模拟真实的按压时间
        state_, reward, done = env.generate_state(action_, state)

        #print(action, state, state_, reward)
        RL.store_transition(state, action, np.float64(reward), state_)

        # 判断是否跳崩了
        if done == True:
            # print(state, reward, done)
            RL.learn()
            state = env.generate_reset_state()
        else:
            state = state_
    print('====================预训练结束====================')
def main(_):
    if FLAGS.pre:
        print('预训练的迭代次数为' + str(FLAGS.pre_epoch))
        pre_train(FLAGS.pre_epoch)
    max_ = 0
    while True:
        state = env.reset()
        tmp = 0
        while True:
            action = RL.choose_action(state)
            # print(action)
            action_ = (action) * 50 + 300
            # action = state * press_coefficient
            state_, reward, done = env.step(action_)
            if not done:
                reward += 0.05 * (tmp + 1)
            RL.store_transition(state, action, np.float64(reward), state_)

            if done == True:
                print('......挂掉了')
                RL.learn()
                env.touch_the_restart()
                break
            tmp +=1
            max_ = max(max_, tmp)
            state = state_
        print('你的阿尔法跳一跳最远跳了:', max_, '下')

env = Env()
if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.variable_scope('dueling'):
            RL = DuelingDQN(
                n_actions=14, n_features=1, memory_size=5000000,
                e_greedy_increment=0.0001, sess=sess, dueling=True, output_graph=True)
        sess.run(tf.global_variables_initializer())
        tf.app.run()