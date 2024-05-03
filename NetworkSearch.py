import networkx as nx
import numpy as np
import random
import queue


#生成BA网络
def generate_BA_network(n, m):
    """
    生成一个具有n个节点的BA网络。
    n: 网络中的节点数
    m: 每次添加新节点时附加的边数
    返回: 生成的BA网络（Graph对象）
    """
    return nx.barabasi_albert_graph(n, m)

def random_walk_to_target(network, s, t):
    """
    执行随机游走从节点s到节点t，并返回首达时间。
    network: 网络（Graph对象）
    s: 源节点
    t: 目标节点
    返回: 首达时间（步数）
    """
    current_node = s
    steps = 0
    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        if t in neighbors:
            return steps + 1  # 如果t是邻居，则立即到达t
        current_node = random.choice(neighbors)  # 否则随机选择一个邻居
        steps += 1
    return steps

def max_degree_search(network, s, t):
    """
    执行最大度搜索策略从节点s到节点t，并返回搜索步数。
    network: 网络（Graph对象）
    s: 源节点
    t: 目标节点
    返回: 搜索步数
    """
    current_node = s
    steps = 0
    visited_edges = set()  # 用于存储已访问的边，确保同一条边不被重复访问
    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        if t in neighbors:
            return steps + 1  # 如果t是邻居，则立即到达t
        # 移除已访问过的邻居节点，避免重复访问同一条边
        neighbors = [n for n in neighbors if (current_node, n) not in visited_edges and (n, current_node) not in visited_edges]
        # 如果当前节点已无未访问的邻居，则无法继续搜索
        if not neighbors:
            return None
        # 选择度数最大的邻居节点，如果有多个则随机选择一个
        max_degree_neighbor = max(neighbors, key=lambda n: network.degree(n))
        visited_edges.add((current_node, max_degree_neighbor))  # 标记这条边为已访问
        current_node = max_degree_neighbor  # 更新当前节点
        steps += 1
    return steps

def min_degree_search(network, s, t):
    current_node = s
    steps = 0
    visited_edges = set()  # 用于存储已访问的边，确保同一条边不被重复访问
    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        if t in neighbors:
            return steps + 1  # 如果t是邻居，则立即到达t
        # 移除已访问过的邻居节点，避免重复访问同一条边
        neighbors = [n for n in neighbors if (current_node, n) not in visited_edges and (n, current_node) not in visited_edges]
        # 如果当前节点已无未访问的邻居，则无法继续搜索
        if not neighbors:
            return None
        # 选择度数最小的邻居节点，如果有多个则随机选择一个
        min_degree_neighbor = min(neighbors, key=lambda n: network.degree(n))
        visited_edges.add((current_node, min_degree_neighbor))  # 标记这条边为已访问
        current_node = min_degree_neighbor  # 更新当前节点
        steps += 1
    return steps

def prioritize(network, s, t):
    """
    执行随机游走从节点s到节点t，并返回首达时间。
    network: 网络（Graph对象）
    s: 源节点
    t: 目标节点
    alpha: 度数的概率权重指数
    返回: 首达时间（步数）
    """
    current_node = s
    steps = 0
    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        if t in neighbors:
            return steps + 1  # 如果t是邻居，则立即到达t
        degrees = np.array([network.degree(n) for n in neighbors])
        probabilities = degrees / np.sum(degrees)
        current_node = np.random.choice(neighbors, p=probabilities)  # 依据概率选择邻居
        steps += 1
    return steps

def prioritize_with_alpha(network, s, t, alpha=1.0):
    """
    执行随机游走从节点s到节点t，并返回首达时间。
    network: 网络（Graph对象）
    s: 源节点
    t: 目标节点
    alpha: 度数的概率权重指数
    返回: 首达时间（步数）
    """
    current_node = s
    steps = 0
    while current_node != t:
        neighbors = list(network.neighbors(current_node))
        if t in neighbors:
            return steps + 1  # 如果t是邻居，则立即到达t
        degrees = np.array([network.degree(n)**alpha for n in neighbors])
        probabilities = degrees / np.sum(degrees)
        current_node = np.random.choice(neighbors, p=probabilities)  # 依据概率选择邻居
        steps += 1
    return steps

if __name__=='__main__':
    for N in [100, 1000, 10000]:
        print(f"N = {N}")
        G = generate_BA_network(N, 5)
        source_node = random.choice(list(G.nodes))  # 随机选择一个源节点
        target_node = random.choice(list(G.nodes))  # 随机选择一个目标节点
        # 确保源节点和目标节点不同
        while source_node == target_node:
            target_node = random.choice(list(G.nodes))
        #随机游走
        # 计算平均首达时间
        mfpt = random_walk_to_target(G, source_node, target_node)
        print(f"随机游走: "+str(mfpt))

        #最大度
        # 执行最大度搜索策略并计算搜索步数
        mfpt = max_degree_search(G, source_node, target_node)
        print(f"最大度搜索策略: "+str(mfpt))

        #最小度
        # 执行最小度搜索策略并计算搜索步数
        mfpt = min_degree_search(G, source_node, target_node)
        print(f"最小度搜索策略: "+str(mfpt))

        #优先附着
        mfpt = prioritize(G, source_node, target_node)
        print(f"优先附着策略: "+str(mfpt))
    
    G = generate_BA_network(10000, 5)
    source_node = random.choice(list(G.nodes))  # 随机选择一个源节点
    target_node = random.choice(list(G.nodes))  # 随机选择一个目标节点
    for alpha in np.arange(0, 1, 0.1):
        mfpt = prioritize_with_alpha(G, source_node, target_node, alpha)
        print(f"alpha={alpha:.2f}: "+str(mfpt))
