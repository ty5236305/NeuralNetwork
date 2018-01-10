# -*- coding: UTF-8 -*-

import random

import numpy as np

class Network(object):

    def __init__(self, sizes):

        self.sizes = sizes
        # 层的数量
        self.num_layers = len(sizes)
        # 所有结点的偏移量，三维数组(每层-每个神经结点-每个偏移值)
        # 除了第一层外所有层都有偏移值
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 所有结点的偏移量，三维数组(每层-每个神经结点-每个权重)
        # 除了第一层外所有层都有权重值
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # 神经结点的激活函数
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # 激活函数的导数
    def sigmoid_prime(self, z):
        a = self.sigmoid(z)
        return a * (1.0 - a)

    # 前向计算
    def feedforward(self, a):
        # 遍历每一层，计算每层的输出值，每层的输出值依赖于上一层的输出
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    # 输出与样本预期结果相同的数量
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y)  for x, y in test_results)

    # 随机梯度下降算法
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            # 对训练数据随机分组
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            # 对训练组内的每条样本计算w和b的梯度，并根据梯度修正w和b
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # 每一代训练结束，用测试集进行测试
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    # 计算每一组的梯度，并更新weights和
    def update_mini_batch(self, mini_batch, eta):

        # 初始化该组w和b的梯度矩阵，与w和b的结构保持一致
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 计算每个样本的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 更新梯度矩阵
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 修正w和b，修正量为该组梯度的平均值
        self.biases = [b - (eta * nb / len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta * nw / len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]

    # 反向传播算法
    def backprop(self, x, y):

        # 初始化该样本w和b的梯度矩阵，与w和b的结构保持一致
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前一层的激活输出，初始值即为样本输入
        activation = x
        # 所有层的激活输出，初始值只包含样本输入
        activations = [x]

        # 所有层的加权输入
        zs = []

        # 先前向计算神经网络，遍历每一层
        for b, w in zip(self.biases, self.weights):
            # 计算加权输入
            z = np.dot(w, activation) + b
            zs.append(z)
            # 计算激活输出
            activation = self.sigmoid(z)
            activations.append(activation)
        # 计算最后一层的误差delta，即代价函数关于加权输入的导数
        # 根据链式法则，等价于代价函数关于输出的导数*激活函数关于加权输入的导数
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        # 偏移的梯度等于误差，
        nabla_b[-1] = delta
        # 权重的梯度等于前一层输出 点乘 当前层的误差
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 从倒数第二层开始向前逐层计算每层的误差
        for l in xrange(2, self.num_layers):
            # 当前层的误差等于后一层的权重 点乘 后一层的误差 * 当前层激活函数关于加权输入的导数
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_prime(zs[-l])
            # 偏移的梯度即误差
            nabla_b[-l] = delta
            # 权重的梯度即误差 点乘 前一层的输出
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    #损失函数的导数
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)



