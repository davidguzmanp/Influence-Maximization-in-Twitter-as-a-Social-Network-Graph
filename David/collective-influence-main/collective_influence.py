# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:57:55 2020

@author: RavindranathNemani
"""

import sys

sys.setrecursionlimit(100**3)

sys.path.append('./')

import numpy as np

import pandas as pd

import heaps

import networkx as nx

#sp = find_shortest_path(graph, vertex, node)
def find_shortest_path(graph, start, end, path=[]):
        #print("start in find_shortest_path---" , start)
        #print("end in find_shortest_path---" , end)

        path = path + [start]

        if start == end:
            #print("The path is..............................", path)
            return path

        if not start in graph.keys():
            return None

        shortest = None

        for node in graph[start]:

            if node not in path:
                #print("node in find_shortest_path---" , node)
                #print("graph[start] in find_shortest_path---", graph[start])
                newpath = find_shortest_path(graph, node, end, path)
                #print(newpath)

                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        #print(shortest)
        #input("___")
        return shortest


def find_nodes_on_frontier(graph, vertex, st, l):

    if l == 0:
        st.add(vertex)

    else:
        st.add(vertex)

        #print(graph[vertex])
        #input("hello")
        for n in graph[vertex]:
            find_nodes_on_frontier(graph, n, st, l-1)

    return st


def get_keys_list(ci):

    l = []

    for key in ci.keys():
        l.append(key)

    return(l)


def find_nodes_on_frontier1(graph, ngraph, vertex, l):

    all_nodes = list(find_nodes_on_frontier(graph, vertex, set(), l))
    #print(all_nodes)
    #input("all_nodes")
    #input("in the ball")

    sp_dists = {}
    for node in all_nodes:
        #input("in the ball___")
        #print("node in fun(find_nodes_on_frontier1)----" , node)
        sp = nx.shortest_path(ngraph, vertex, node)
        #print(sp)
        #input("shortest path is............................")
        #sp = find_shortest_path(graph, vertex, node)
        #sp.remove(vertex)
        #print("node")
        #print(node)
        #print("sp")
        #print(len(sp))
        #input("uday")
        sp_dists[node] = len(sp)

    for key in sp_dists.copy():
        if sp_dists[key] < l+1:
            del sp_dists[key]

    return get_keys_list(sp_dists)


def mean(d):

    mean = 0.0

    for key in d.keys():
        mean = mean + d[key]

    mean = mean / len(d)

    return(mean)


def get_degrees(graph):

    degrees = graph.copy()

    for key in graph.keys():
        degrees[key] = len(graph[key])

    return degrees


def get_reduced_degrees(graph):

    degrees = graph.copy()

    for key in graph.keys():
        degrees[key] = len(graph[key]) - 1

    return degrees


def collective_influence(graph, ngraph, l):

    degs = get_reduced_degrees(graph)
    #print(degs)
    #input("test0")

    ci = graph.copy()
    #print(graph)
    #input("rtest")
    #i = 0
    for vertex in graph.keys():
        #print(vertex)
        #input("vertex")
        neighborhood = find_nodes_on_frontier1(graph, ngraph, vertex, l)
        #print(neighborhood)
        neighborhood = list(neighborhood)
        if vertex in neighborhood:
            neighborhood.remove(vertex)
        #print(neighborhood)
        #input("neighborhood")

        ci_vertex = degs[vertex]
        su = 0.0

        for node in range(len(neighborhood)):
            #print(node)
            #print(neighborhood[node])
            #print(degs)
            #print(degs[neighborhood[node]])
            #input("test1")
            su = su + degs[neighborhood[node]]

        ci_vertex = int(ci_vertex * su)
        ci[vertex] = ci_vertex
        #print(i)
        #i = i+1
    return(ci)


def approximate_largest_eigenvalue(graph, l, ci):

    #ci = collective_influence(graph)

    k = get_degrees(graph)

    lambd = mean(ci) / mean(k)
    #print(l)
    #input("exponent")
    exponent = 1 / (l+1)
    lambd = pow(lambd, exponent)

    return(lambd)


def get_list(ci):

    l = []

    for key in ci.keys():
        l.append(ci[key])

    return(l)


def delete_vertex(graph, vertex):

    if vertex in graph.keys():
        del graph[vertex]
        #print(graph)

    for key in graph.keys():
        if vertex in graph[key]:
            graph[key].remove(vertex)

    return(graph)


def get_vertex(ci, maximum_ci):
    for key in ci.keys():
        if ci[key] == maximum_ci:
            break

    return key


def update_values(graph, l, ci):

    #ci = collective_influence(graph, ngraph, l)

    ev = approximate_largest_eigenvalue(graph, l, ci)

    lst = get_list(ci)

    max_heap = heaps.MaxHeap(lst)
    #print(type(max_heap))
    #input("test0")

    return max_heap, ev


def collective_influence_only_in_ball(graph1, ngraph1, vertex, l, ci):

    degs = get_reduced_degrees(graph1)
    #print(vertex)
    #input("ravi")

    neighborhood = set()

    for j in range(l+2):
        if j > 0:
            neighborhood.update(find_nodes_on_frontier1(graph1, ngraph1, vertex, j))

    delete_vertex(graph1, vertex)
    ngraph1 = nx.from_dict_of_lists(graph1)

    del ci[vertex]

    for node in neighborhood:
        node_neighborhood = find_nodes_on_frontier1(graph1, ngraph1, node, l)

        node_neighborhood = list(node_neighborhood)
        if node in node_neighborhood:
            node_neighborhood.remove(node)

        ci_node = degs[node]

        su = 0.0

        for n in range(len(node_neighborhood)):
            #print(node)
            #print(neighborhood[node])
            #print(degs)
            #print(degs[neighborhood[node]])
            #input("test1")
            su = su + degs[node_neighborhood[n]]

        ci_node = int(ci_node * su)
        ci[node] = ci_node

    return(ci)


def get_influencers(graph, ngraph, l):

    ci = collective_influence(graph, ngraph, l)
    #print(ci)
    #input("test")
    lst = get_list(ci)

    max_heap = heaps.MaxHeap(lst)

    #max_heap = heaps.MaxHeap.get_list(max_heap)

    ev = approximate_largest_eigenvalue(graph, l, ci)

    i = 1

    removed_vertices = []

    while ev > 1:

        #create a copy
        #graph1 = graph.copy()

        #heapify
        #max_heap = heaps.MaxHeap.get_list(max_heap)
        lst1 = get_list(ci)

        max_heap = heaps.MaxHeap(lst1)

        #extract_max
        maximum_ci = heaps.MaxHeap.extract_max(max_heap)
        #print(maximum_ci)
        #input("maximum")
        #print(maximum_ci)
        #input("test1")
        #delete root in heap
        #heaps.MaxHeap.delete()

        #locate the vertex which has maximum_ci as it's maximum value
        vertex = get_vertex(ci, maximum_ci)
        #print(ci)
        #input("ci")
        #print(vertex)
        #input("test2")

        #create copies of original graph before deleting the vertex
        graph1 = graph.copy()
        ngraph1 = ngraph.copy()

        #also delete corresponding node from the graph
        delete_vertex(graph, vertex)
        ngraph = nx.from_dict_of_lists(graph)

        #heapify
        max_heap = heaps.MaxHeap.get_list(max_heap)

        #update CI values
        #max_heap, ev = update_values(graph, l)

        #heapify
        #max_heap = heaps.MaxHeap.get_list(max_heap)

        #compute new ci values
        #ci = collective_influence(graph, ngraph, l)
        #change CI values only within ball of radius l+1
        ci = collective_influence_only_in_ball(graph1, ngraph1, vertex, l, ci)
        max_heap, ev = update_values(graph, l, ci)
        #compute new largest eigenvalue
        ev = approximate_largest_eigenvalue(graph, l, ci)

        removed_vertices.append(vertex)

        #print(ev)
        #print(i)
        #input("test3")

        i = i+1

    return(removed_vertices)
'''
graph = { 1 : [2, 3],
      2 : [1, 4, 5, 6],
      3 : [1, 4],
      4 : [2, 3, 5],
      5 : [2, 4, 6],
      6 : [2, 5]
    }

ci = collective_influence(graph, 1)

l = get_list(ci)

max_heap = heaps.MaxHeap(l)
l1 = heaps.MaxHeap.get_list(max_heap)
print("Hello")
print(type(l1))
print(l1)
'''

def get_graph():

    edges = [1,4, 2,4, 3,4, 3,5, 4,5, 4,7, 5,6, 5,29, 6,7, 6,10, 6,20, 7,8, 7,9, 7,10, 10,11, 10,30, 11,12, 11,17, 12,13, 12,14, 12,15, 12,16, 14,15, 16,17, 17,18, 17,29, 18,19, 18,20, 20,21, 21,22, 21,23, 21,24, 21,27, 21,30, 22,23, 24,25, 24,26, 24,27, 24,29, 26,27, 27,28, 27,29]

    num_edges = int(len(edges)/2)

    vertices = list(np.unique(edges))

    #add vertices

    graph = {}

    for i in range(len(vertices)):

        graph[vertices[i]] = []

    for j in range(len(vertices)):

        for k in range(num_edges):

            if vertices[j] == edges[2*k+1]:
                graph[vertices[j]].append(edges[2*k])

            elif vertices[j] == edges[2*k]:
                graph[vertices[j]].append(edges[2*k+1])

    return(graph)


def get_data():

    df = pd.read_csv('C:/Users/RavindranathNemani/Desktop/DeepLearning/R/Sanjeev/Relation/data.txt', sep = "\t", header = None, names = ["vertex", "edges"])

    graph = {}

    for i in range(len(df)):

        edge_string = df.loc[i, "edges"]
        edge_list = edge_string.split(',')

        graph[df.loc[i, "vertex"]] = edge_list

        #print(edge_string)
        #input("edge_string")
    return(graph)


def graph_check(graph):
    #i = 0
    vertex_list_from_edges = []
    for vertex in graph.keys():
        #print(vertex)
        #print(i)
        #i = i+1
        vertex_list_from_edges = vertex_list_from_edges + graph[vertex]

    vertex_list_from_edges = np.unique(vertex_list_from_edges)
    #vertex_list_from_edges.sort()
    vertex_list_from_edges = list(vertex_list_from_edges)
    vertex_list_from_edges = set(vertex_list_from_edges)

    vertex_list = get_keys_list(graph)
    #vertex_list.sort()
    vertex_list = set(vertex_list)

    #print(type(vertex_list))
    #input("original")
    #print(type(vertex_list_from_edges))
    #input("from edges")

    if vertex_list == vertex_list_from_edges:
        return True
    else:
        return False



'''
sp = find_shortest_path(graph, 3, 21)
print(sp)
input("sp")

ngbhd = find_nodes_on_frontier1(graph, 3, 3)
print(ngbhd)
input("ravi")
'''
graph = get_data()
#print(graph)
ngraph = nx.from_dict_of_lists(graph)
'''
if graph_check(graph):
    print("Graph is properly structured")
    #input("correct")
    removed_vertices = get_influencers(graph, ngraph, 3)
else:
    print("Graph not properly structured")
    #input("npot correct")
'''
removed_vertices = get_influencers(graph, ngraph, 3)
print(removed_vertices)