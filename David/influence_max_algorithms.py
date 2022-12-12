from copy import deepcopy
import networkx as ntwx
from tqdm import tqdm

class MIA():
# greatly optimized version of the python implementation proposed by https://github.com/ksasi/fairMIA
    def __init__(self, graph) -> None:
        self.mir_pairs_global = dict(ntwx.all_pairs_dijkstra(graph, weight='weight_negative_log'))

    def getpp(self, network, w, u):
        """Returns path propagation probability"""
        pp = 1
        #path = ntwx.dijkstra_path(network, w, u, weight = 'weight')
        path = self.mir_pairs_global[w][1][u]
        edges = list(ntwx.utils.misc.pairwise(path))
        for edge in edges:
            #print(network[edge[0]][edge[1]]['weight'])
            pp = pp*network[edge[0]][edge[1]]['weight']
        return pp


    def getMIIA(self, graph, v, theta, digraph = True):
        "Returns MAXIMUM INFLUENCE IN-ARBORESCENCE"
        #nodes = list(dgraph.nodes())
        #print('hello getMIIA')
        if digraph == True:
            miia = ntwx.DiGraph()
        else:
            miia = ntwx.Graph()

        #mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight')) 
        mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight_negative_log'))


        for val in mir_pairs.values():
            #print(val[1].values())
            if v in val[1].keys():
                #print(v)
                pp = self.getpp(graph, val[1][v][0], val[1][v][-1])
                if pp > theta:
                    edges = list(ntwx.utils.misc.pairwise(val[1][v]))
                    miia.add_edges_from(edges)
                    #miia.add_nodes_from(val[1][v])
                    #ntwx.draw(miia)
        for edge in miia.edges():
         miia[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
        #print('goodbye getMIIA')
        return miia

    def getMIIA_global(self, graph, v, theta, digraph = True):
        "Returns MAXIMUM INFLUENCE IN-ARBORESCENCE"
        #nodes = list(dgraph.nodes())
        #print('hello getMIIA')
        if digraph == True:
            miia = ntwx.DiGraph()
        else:
            miia = ntwx.Graph()
            
        #mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight')) 
        
        for val in self.mir_pairs_global.values():
            #print(val[1].values())
            if v in val[1].keys():
                #print(v)
                pp = self.getpp(graph, val[1][v][0], val[1][v][-1])
                if pp > theta:
                    edges = list(ntwx.utils.misc.pairwise(val[1][v]))
                    miia.add_edges_from(edges)
                    #miia.add_nodes_from(val[1][v])
                    #ntwx.draw(miia)
    
        for edge in miia.edges():
            miia[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
        #print('goodbye getMIIA')
    
        return miia

    def getMIOA(self, graph, v, theta, digraph = True):
        "Returns MAXIMUM INFLUENCE OUT-ARBORESCENCE"
        if digraph == True:
            mioa = ntwx.DiGraph()
        else:
            mioa = ntwx.Graph()

        #mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight'))
        mir_pairs = dict(ntwx.all_pairs_dijkstra(graph, weight='weight_negative_log'))


        if v in mir_pairs.keys():
            for val in mir_pairs[v][1].values():
                pp = self.getpp(graph, val[0], val[-1])
                if pp > theta:
                    #print(val)
                    edges = list(ntwx.utils.misc.pairwise(val))
                    #print(edges)
                    mioa.add_edges_from(edges)
                    #ntwx.draw(mioa)
        for edge in mioa.edges():
            mioa[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight']
        return mioa


    def getap(self, u, S, MIIA): #problematic
        """Returns activation probability given a note and MIIA"""
        #print('hello getap')
        if len(S) == 0:
            return 0
        if u in S :
            return 1
        elif len(list(MIIA.predecessors(u))) == 0:
            return 0
        else:
            ap = 1
            #print(list(MIIA.predecessors(u)))
            for w in list(MIIA.predecessors(u)):
                #print("Node:", w)
                ap = 1 - ap*(1 - self.getap(w, S, MIIA)*self.getpp(MIIA, w, u))
                #print(ap)
                #print("\n")
        #print('goodbye getap')
        return ap

    def getallap(self, S, MIIA):
        """Returns a dictionary activation probabilities for all nodes in MIIA"""
        ap_dict = dict()
        node_nb_dict = dict()
        for n in MIIA.nodes():
            node_nb_dict[n] = len(list(MIIA.predecessors(n)))
        #temp = list(ntwx.topological_sort(MIIA))
        #print(temp)
        #print(node_nb_dict)
        #print(sorted(node_nb_dict, key=node_nb_dict.get))
        for u in list(ntwx.topological_sort(MIIA)):
            if u in S :
                ap_dict[u] = 1
            elif node_nb_dict[u] == 0:
                ap_dict[u] = 0
            else:
                ap = 1
                #print(list(MIIA.predecessors(u)))
                #print(u)
                #print(list(MIIA.predecessors(u)))
                for w in list(MIIA.predecessors(u)):
                    #print("\n")
                    #print("Node:", w)
                    #print(node_nb_dict[w])
                    ap = ap*(1 - ap_dict[w]*self.getpp(MIIA, w, u))
                    #print(ap)
                    #print("\n")
                ap_dict[u] = 1 - ap
        return ap_dict


    def getalpha(self, v, u, S, network, theta): #super problematic
        """ Returns the value of aplha as per Algorithm3"""
        #print('hello getalpha')
        MIIA = self.getMIIA(network, v, theta, True)
        if v == u :
            return 1
        else:
            w = list(MIIA.successors(u))[0]
            if w in S :
                return 0
            else:
                alpha = self.getalpha(v, w, S, MIIA, theta)*self.getpp(MIIA, u, w)
                for u_dash in list(MIIA.predecessors(w)):
                    alpha = alpha * (1 - self.getap(u_dash, S, MIIA)*self.getpp(MIIA, u_dash, w))
        #print('goodbye getalpha')
        return alpha


    def MIA_fast(self, network, k, theta):

        print("Starting MIA intialization")

        S = []
        Incinf_dict = dict()
        ap_dict = dict()
        for v in list(network.nodes()):
            Incinf_dict[v] = 0
        for v in tqdm(list(network.nodes())):
            MIIAv = self.getMIIA_global(network, v, theta, digraph = True)
            MIIA = MIIAv
            #MIOA = getMIOA(network, v, theta, digraph = True)
            #print(len(list(MIIA.nodes())))
            for u in list(MIIA.nodes()):
                Incinf_dict[u] = Incinf_dict[u] + self.getalpha(v, u, S, MIIAv, theta) * (1 - self.getap(u, S, MIIA))

        print("Initialization Completed")

        for i in tqdm(range(1, k+1)):

            #from Incinf_dict remove the nodes in S
            Incinf_dict_for_S = Incinf_dict.copy()
            for v in S:
                Incinf_dict_for_S.pop(v)
            
            u = max(Incinf_dict_for_S, key = Incinf_dict_for_S.get)
            MIOA = self.getMIOA(network, u, theta, digraph = True)
            for v in list(MIOA.nodes()):
                MIIAv = self.getMIIA_global(network, v, theta, digraph = True)
                ap_all = self.getallap(S, MIIAv)
                for w in list(MIIAv.nodes()):
                    Incinf_dict[w] = Incinf_dict[w] - self.getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
            S.append(u)
            #print(S)
            for v in list(MIOA.nodes()): 
                MIIAv = self.getMIIA_global(network, v, theta, digraph = True)
                ap_all = self.getallap(S, MIIAv)
                for w in list(MIIAv.nodes()):
                    Incinf_dict[w] = Incinf_dict[w] + self.getalpha(v, w, S, MIIAv, theta)*(1 - ap_all[w])
        return S

    def initial_incinf(self, network, theta):

        print("Starting MIA intialization")

        S = []
        Incinf_dict = dict()
        ap_dict = dict()
        for v in list(network.nodes()):
            Incinf_dict[v] = 0
        for v in tqdm(list(network.nodes())):
            MIIAv = self.getMIIA_global(network, v, theta, digraph = True)
            MIIA = MIIAv
            #MIOA = getMIOA(network, v, theta, digraph = True)
            #print(len(list(MIIA.nodes())))
            for u in list(MIIA.nodes()):
                Incinf_dict[u] = Incinf_dict[u] + self.getalpha(v, u, S, MIIAv, theta) * (1 - self.getap(u, S, MIIA))

        return Incinf_dict
