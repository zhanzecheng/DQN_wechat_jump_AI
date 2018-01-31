# Q_Learning 微信跳一跳
将 DQN 应用在微信跳一跳小程序
在游戏中，把棋子中心和方块中心的距离作状态(state)，按压屏幕的时间作为action来做强化学习。

## 效果展示

不是很稳定...
![image](https://github.com/zhanzecheng/DQN_wechat_jump_AI/blob/master/data/demo.gif)

## 预训练
由于正常游戏中样本量太少，这里采用了预训练的方式来加速强化学习的过程。每次随机生成距离状态量，根据预设的参数来求出虚拟的真实按压时间。利用它与模型输出的action做比较，判断游戏是否game over。
## Requirement

python3  
tensorflow > 1.2.0  
tqdm

## Installing 

#### tensorflow
    sudo pip3 install tensorflow
    
#### tqdm
    sudo pip3 install tqdm

#### ADB 
使用ADB来从安卓机中截图，如果使用的是mac电脑，安装如下：

    brew cask install android-platform-tools
手机USB连接上电脑之后，终端输入命令

    adb devices
如果看到显示信息是

    List of devices attached
    6934dc33    device
便完成配置 . 
windows和linux环境配置可参考下面的连接
[配置链接](https://github.com/wangshub/wechat_jump_game/wiki/Android-%E5%92%8C-iOS-%E6%93%8D%E4%BD%9C%E6%AD%A5%E9%AA%A4)


## RUN

微信打开跳一跳游戏界面
    
    git clone https://github.com/zhanzecheng/DQN_wechat_jump_AI.git
    cd src
    python3 run_this.py

默认使用了`预训练模式`
可以通过参数选择是否使用预训练模式和预训练的次数

    python3 run_this.py --pre=True --pre_epoch=100000
    
config文件夹包含这各种手机的屏幕参数信息，若效果不好，请根据需要调节。

## TODO
1. 开发ios版本：正在进行中
2. 优化reward函数：等待
3. 采用连续的action网络函数来替代DQN：等待
