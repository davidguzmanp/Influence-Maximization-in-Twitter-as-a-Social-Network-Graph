# Non-backtracking centrality using scipy.sparse and networkx
# Assumes nodes in G are indexed by integers (0,1,...,n-1), equivalently:
# G = nx.convert_node_labels_to_integers(G_original)
# -GTC, 6/9/22
import numpy as np
import scipy.sparse
import networkx as nx

def IharaMatrix(G):
    '''
    Return the matrix
    M = ( A  I-D )
        ( I   0  )
    as a sparse matrix
    '''
    n,m = G.number_of_nodes(), G.number_of_edges()
    a,b,c = np.zeros(2*(m+n)), np.zeros(2*(m+n)), np.zeros(2*(m+n))
    l = 0
    for i,j in G.edges():
        a[l],b[l],c[l] = i,j,1
        a[l+1],b[l+1],c[l+1] = j,i,1
        l += 2
    for i in G.nodes():
        a[l],b[l],c[l] = i,n+i,1-G.degree(i)
        a[l+1],b[l+1],c[l+1] = n+i,i,1
        l += 2
    return scipy.sparse.csr_matrix((c, (a,b)), shape=(2*n,2*n))

def non_backtracking_centrality(G):
    M = IharaMatrix(G)
    v = scipy.sparse.linalg.eigs(M,k=1)[1][:G.number_of_nodes()]
    v = np.real(v.T)[0]
    v = v*np.sign(np.sum(v))
    return v

