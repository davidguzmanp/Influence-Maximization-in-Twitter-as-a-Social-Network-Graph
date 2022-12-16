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

class Decreasing_Cascade():
    def __init__(self):
        self.g = nx.DiGraph()
        self.num_nodes = 0
        self.node_label = []
        self.label2id = {}
        self.max_in_degree = 0
        self.probability = None

    def fit(self, g):
        # fit graph with probability
        self.g = g
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        self.max_in_degree = max(j for _, j in g.in_degree(weight='None'))
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
        inform_times = np.zeros(self.num_nodes)
        while(True):
            new_activated_nodes = []
            new_inform_times = np.zeros(self.num_nodes)
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    new_inform_times[neighbor] += 1
                    if (self.probability[node][neighbor] >= (rd.random()+ inform_times[neighbor] / self.max_in_degree) ):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
            inform_times = inform_times + new_inform_times
        return activated_nodes, activate_nums

    # diffusion to max step
    def diffusion_step(self, seed_nodes, max_step=1):
        if(seed_nodes == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in seed_nodes]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        activate_nums = [len(activated_nodes)]
        inform_times = np.zeros(self.num_nodes)
        for step in range(max_step):
            new_activated_nodes = []
            new_inform_times = np.zeros(self.num_nodes)
            for node in old_activated_nodes:
                neighbors = self.probability[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    if(neighbor in activated_nodes and neighbor not in new_activated_nodes):
                        continue
                    new_inform_times[neighbor] += 1
                    if (self.probability[node][neighbor] >= (rd.random()+ inform_times[neighbor] / self.max_in_degree) ):
                        new_activated_nodes.append(neighbor)
            activated_nodes.extend(new_activated_nodes)
            if len(new_activated_nodes) == 0:
                break
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            activate_nums.append(len(new_activated_nodes))
            inform_times = inform_times + new_inform_times
        return activated_nodes, activate_nums

class linear_threshold():
    def __init__(self):
        self.g = nx.DiGraph()
        self.influence = None
    
    def fit(self, g):
        # fit graph with probability
        in_degree = g.in_degree()
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        max_degree = max([d for (n, d) in in_degree])
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        self.influence = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        # init influence 
        for e in g.edges():
            self.influence[self.label2id[e[0]], self.label2id[e[1]]] = 1 / in_degree[e[1]]
        self.g = g
        return g
    
    # diffusion to all possible nodes
    def diffusion_all(self, init_seed, threshold=0.001):
        if(init_seed == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in init_seed]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        # the index represent time t, the value is number of activated nodes in current time
        times = [len(old_activated_nodes)]
        while True:
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.influence[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    total_influence = 0.0
                    if(neighbor in activated_nodes or neighbor in new_activated_nodes):
                        continue
                    precessors = self.influence[:, neighbor].nonzero()[0]
                    for precessor in precessors:
                        if precessor in activated_nodes:
                            total_influence += self.influence[precessor][neighbor]
                    if(total_influence >= threshold):
                        new_activated_nodes.append(neighbor)
            if len(new_activated_nodes)==0:
                break
            else:
                activated_nodes.extend(new_activated_nodes)
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            times.append(len(new_activated_nodes))
        return activated_nodes, times
    
    # diffusion to max step
    def diffusion_step(self, init_seed, threshold=0.1, max_step=1):
        if(init_seed == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in init_seed]
        old_activated_nodes = copy.deepcopy(activated_nodes)
        # the index represent time t, the value is number of activated nodes in current time
        times = [len(old_activated_nodes)]
        for _ in range(max_step):
            new_activated_nodes = []
            for node in old_activated_nodes:
                neighbors = self.influence[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    total_influence = 0.0
                    if(neighbor in activated_nodes or neighbor in new_activated_nodes):
                        continue
                    precessors = self.influence[:, neighbor].nonzero()[0]
                    for precessor in precessors:
                        if precessor in activated_nodes:
                            total_influence += self.influence[precessor][neighbor]
                    if(total_influence >= threshold):
                        new_activated_nodes.append(neighbor)
            if len(new_activated_nodes)==0:
                break
            else:
                activated_nodes.extend(new_activated_nodes)
            old_activated_nodes = copy.deepcopy(new_activated_nodes)
            times.append(len(new_activated_nodes))
        return activated_nodes, times
    
    
class general_threshold():
    def __init__(self):
        self.g = nx.DiGraph()
        self.influence = None
        # threshold determine whether the node become activated
        self.threshold = None
        # spreadTrd determine whether the node have the ability to spread message
        self.spreadTrd = None
    
    def fit(self, g):
        # fit graph with probability
        in_degree = g.in_degree()
        self.num_nodes = g.number_of_nodes()
        self.node_label = [i for i in g.nodes()]
        self.label2id = {self.node_label[i]: i for i in range(self.num_nodes)}
        self.influence = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        centrality = nx.degree_centrality(g)
        # init influence
        for e in g.edges():
            self.influence[self.label2id[e[0]], self.label2id[e[1]]] = 1 / in_degree[e[1]]
        # init threshold. In GT model, the threshold of each node is assigned half of it's centrality
        self.threshold = np.zeros((self.num_nodes), dtype=float)
        self.spreadTrd = np.zeros((self.num_nodes), dtype=float)
        for n in g.nodes():
            self.threshold[self.label2id[n]] = centrality[n]/4
            self.spreadTrd[self.label2id[n]] = centrality[n]/2
        self.g = g
        return g
    
    # diffusion to all possible nodes
    def diffusion_all(self, init_seed):
        if(init_seed == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in init_seed]
        old_spread_nodes = copy.deepcopy(activated_nodes)
        # the index represent time t, the value is number of activated nodes in current time
        times = [len(old_spread_nodes)]
        while True:
            new_activated_nodes = []
            new_spread_nodes = []
            for node in old_spread_nodes:
                neighbors = self.influence[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    total_influence = 0.0
                    if(neighbor in activated_nodes or neighbor in new_activated_nodes):
                        continue
                    precessors = self.influence[:, neighbor].nonzero()[0]
                    for precessor in precessors :
                        if precessor in activated_nodes:
                            total_influence += self.influence[precessor][neighbor]
                    # informe/activate the node if the total influence exceeds it's threshold of activated
                    if(total_influence >= self.threshold[neighbor]):
                        new_activated_nodes.append(neighbor)
                        # let node be a spreader if the total influence exceeds it's threshold of spread
                        if(total_influence >= self.spreadTrd[neighbor]):
                            new_spread_nodes.append(neighbor)
            if len(new_spread_nodes)==0:
                break
            else:
                activated_nodes.extend(new_activated_nodes)
            old_spread_nodes = copy.deepcopy(new_spread_nodes)
            times.append(len(new_activated_nodes))
        return activated_nodes, times
    
    # diffusion to max step
    def diffusion_step(self, init_seed, max_step=1):
        if(init_seed == []):
            return [], []
        activated_nodes = [self.label2id[name] for name in init_seed]
        old_spread_nodes = copy.deepcopy(activated_nodes)
        # the index represent time t, the value is number of activated nodes in current time
        times = [len(old_spread_nodes)]
        for _ in range(max_step):
            new_activated_nodes = []
            new_spread_nodes = []
            for node in old_spread_nodes:
                neighbors = self.influence[node, :].nonzero()[0]
                if(len(neighbors) == 0):
                    continue
                for neighbor in neighbors:
                    total_influence = 0.0
                    if(neighbor in activated_nodes or neighbor in new_activated_nodes):
                        continue
                    precessors = self.influence[:, neighbor].nonzero()[0]
                    for precessor in precessors:
                        if precessor in activated_nodes:
                            total_influence += self.influence[precessor][neighbor]
                    # informe/activate the node if the total influence exceeds it's threshold of activated
                    if(total_influence >= self.threshold[neighbor]):
                        new_activated_nodes.append(neighbor)
                        # let node be a spreader if the total influence exceeds it's threshold of spread
                        if(total_influence >= self.spreadTrd[neighbor]):
                            new_spread_nodes.append(neighbor)
            if len(new_spread_nodes)==0:
                break
            else:
                activated_nodes.extend(new_activated_nodes)
            old_spread_nodes = copy.deepcopy(new_spread_nodes)
            times.append(len(new_activated_nodes))
        return activated_nodes, times
    


class Trivalency_Model_Higher_Prob():
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
        p_list = [0.5, 0.3, 0.1]
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