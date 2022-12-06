import networkx as nx
import numpy as np
import random as rd
import copy


class Weighted_Cascade():
    def __init__(self):
        self.g = nx.DiGraph()
        self.num_nodes = 0
        self.node_label = []
        self.label2id = {}
        self.probability = None

    def fit(self, g):
        self.g = g
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        in_degree = g.in_degree(weight='None')
        self.probability = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        for e in g.edges():
            self.probability[self.label2id[e[0]], self.label2id[e[1]]] = 1 / in_degree[e[1]]
        
    def monte_carlo_diffusion_all(self, seed_nodes, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_all(seed_nodes)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)

    def monte_carlo_diffusion_step(self, seed_nodes, max_step=1, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_step(seed_nodes, max_step)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)
    
    # diffusion to all possible nodes
    def diffusion_all(self, seed_nodes):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        while(True):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums

    # diffusion to max step
    def diffusion_step(self, seed_nodes, max_step=1):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        for step in range(max_step):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums

class Independent_Cascade():
    def __init__(self):
        self.g = nx.DiGraph()
        self.num_nodes = 0
        self.node_label = []
        self.label2id = {}
        self.probability = None

    def fit(self, g):
        self.g = g
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        in_degree = g.in_degree(weight='None')
        self.probability = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        for e in g.edges():
            if(in_degree[e[1]] >= 10):
                self.probability[self.label2id[e[0]], self.label2id[e[1]]] = 1 / int(np.log(in_degree[e[1]]))
            else:
                self.probability[self.label2id[e[0]], self.label2id[e[1]]] = 1
        
    def monte_carlo_diffusion_all(self, seed_nodes, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_all(seed_nodes)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)

    def monte_carlo_diffusion_step(self, seed_nodes, max_step=1, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_step(seed_nodes, max_step)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)
    
    # diffusion to all possible nodes
    def diffusion_all(self, seed_nodes):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        while(True):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums

    # diffusion to max step
    def diffusion_step(self, seed_nodes, max_step=1):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        for step in range(max_step):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums

class Trivalency_Model():
    def __init__(self):
        self.g = nx.DiGraph()
        self.num_nodes = 0
        self.node_label = []
        self.label2id = {}
        self.probability = None

    def fit(self, g):
        self.g = g
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        in_degree = g.in_degree(weight='None')
        self.probability = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        p_list = [0.1, 0.01, 0.001]
        for e in g.edges():
            self.probability[self.label2id[e[0]], self.label2id[e[1]]] = rd.choice(p_list)

        
    def monte_carlo_diffusion_all(self, seed_nodes, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_all(seed_nodes)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)

    def monte_carlo_diffusion_step(self, seed_nodes, max_step=1, num_simulations=100):
        if(seed_nodes == []):
            return [], []
        activate_nums_list = []
        for _ in range(num_simulations):
            _, activate_nums = self.diffusion_step(seed_nodes, max_step)
            activate_nums_list.append(activate_nums)
        narry = np.zeros([len(activate_nums_list),len(max(activate_nums_list,key = lambda x: len(x)))])
        for i,j in enumerate(activate_nums_list):
            narry[i][0:len(j)] = j
        return np.mean(narry, axis=0)
    
    # diffusion to all possible nodes
    def diffusion_all(self, seed_nodes):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        while(True):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums

    # diffusion to max step
    def diffusion_step(self, seed_nodes, max_step=1):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        for step in range(max_step):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    if (self.probability[node][neighbor] >= rd.random()):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
        return activated_nodes, activate_nums