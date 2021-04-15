import numpy as np
from collections import Counter


def gym_preprocess(image):
    image = image[34:194, :, :]  # 160, 160, 3
    image = np.mean(image, axis=2, keepdims=False)  # 160, 160
    image = image[::2, ::2]  # 80, 80
    image = image/256
    # constant = Counter(image).most_common()[0][0]  # constant, eg: 90
    image = image - 90 / 256  # 80, 80
    return image


class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.nodeVal = [0 for _ in range(2 * capacity - 1)]

    def retrieve(self, num):
        """
        :param num:  需要索引的数值，根据这个数值返回index
        :return: 返回叶子结点的index
        """
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if num > self.nodeVal[left]:
                num -= self.nodeVal[left]
                index = right
            else:
                index = left
        return index - (self.capacity - 1)

    def update(self, delta, index):
        """
        :param delta: 需要更新到叶子结点的TD-error
        :param index: 对应叶子节点在SumTree中的index
        :return: 更新self.nodeVal,维持好这个SumTree
        """
        index += self.capacity - 1
        while True:
            self.nodeVal[index] += delta
            if index == 0:
                break
            index -= 1
            index //= 2
