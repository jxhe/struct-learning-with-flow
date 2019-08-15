# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

import pydot


class Graph:

    def __init__(self, nodes, edges):
        """For instance Graph([1,2,3], [(1,2),(2,3),(3,1)]).
        """
        self.nodes = nodes
        self.edges = edges

    def get_node(self):
        """Return any node of the graph.
        """
        return self.nodes[0]

    def nodes_list(self):
        return self.nodes

    def edge(self, n1, n2):
        return (n1, n2) in self.edges or (n2, n1) in self.edges
    
    def neighbors(self, n):
        """Return a list with the neighbors of n.
        """
        return [p for (p,q) in self.edges if q==n] \
            + [q for (p,q) in self.edges if p==n]
    
    def remove_node(self, n):
        if n in self.nodes:
            self.nodes.remove(n)
            for (p, q) in self.edges:
                if n in [p,q]:
                    self.edges.remove((p,q))
    
    def has_cicle(self):
        """assert 'self must be connected'
        Returns True if there is a cicle, False otherwise.
        """
        #if len(self.edges) >= len(self.nodes):
        #    return True
        n = self.get_node()
        stack = [n]
        visited = set()
        while stack != []:
            n = stack.pop()
            if n in visited:
                return True
            else:
                visited.add(n)
                stack += [m for m in self.neighbors(n) if m not in visited]

        return False
    
    def prune_subtrees(self, iterations=-1):
        """Iteratively remove the nodes of degree 1. iterations == -1 means to
        do this until there aren't nodes to remove.
        """
        nodes1 = [n for n in self.nodes_list() if len(self.neighbors(n)) == 1]
        i = 0
        while nodes1 != [] and i != iterations:
            for n in nodes1:
                self.remove_node(n)
            nodes1 = [n for n in self.nodes_list() if len(self.neighbors(n)) == 1]
            i += 1
    
    def compute_connected_components(self):
        """Compute the connected components of the graph in the variable self.ccs.
        """
        nodes = self.nodes_list()
        if not nodes:
            self.ccs = []
            return 0
        #nodes = self.nodes
        ccs = {0: [nodes[0]]}
        nextcc = 1
        for i in range(1, len(nodes)):
            node = nodes[i]
            l = []
            for j in ccs:
                if any(self.edge(node, m) for m in ccs[j]):
                    l += [j]
            if l == []:
                # node is a new cc.
                ccs[nextcc] = [node]
                nextcc += 1
            else:
                # the node joins all the ccs in l.
                cc = ccs[l[0]]
                for j in l[1:]:
                    cc += ccs[j]
                    del ccs[j]
                cc += [node]

        self.ccs = list(ccs.values())
        #return len(self.ccs)

    def is_independent_set(self, nodes, show=False):
        """Checks if a list of nodes is an independent set. If show=True and the
        result is False, prints the first counter-example found.
        """
        #return all(not self.edge(m, n) for m in nodes for n in nodes)
        for m in nodes:
            for n in nodes:
                if self.edge(m, n):
                    if show:
                        print("Edge found:", (m, n))
                    return False
        return True
    
    def get_dot_graph(self, nodes=None):
        if nodes == None:
            nodes = self.nodes
        g = pydot.Dot()
        g.set_type('graph')
        for i in range(1, len(nodes)):
            node = nodes[i]
            for j in range(i):
                # maybe:
                #if self.edge(str(node), str(nodes[j])):
                if self.edge(node, nodes[j]):
                    e = pydot.Edge(node, nodes[j])
                    g.add_edge(e)
        return g
    
    def draw_graph(self, filename, nodes=None):
        """Draw the graph in a JPG file.
        """
        g = self.get_dot_graph(nodes)
        g.write_jpeg(filename, prog='dot')


class WGraph(Graph):
    """Graph with weighted edges.
    """

    def edge_weight(self, n1, n2):
        return 0
    
    def compute_connected_components(self, w_min=1):
        """Compute the connected components of the graph in the variable self.ccs.
        w_min is the minimal weight for the edges to consider.
        """
        nodes = self.nodes_list()
        if not nodes:
            self.ccs = []
            return 0
        #nodes = self.nodes
        ccs = {0: [nodes[0]]}
        nextcc = 1
        for i in range(1, len(nodes)):
            node = nodes[i]
            l = []
            for j in ccs:
                if any(self.edge_weight(node, m) >= w_min for m in ccs[j]):
                    l += [j]
            if l == []:
                # node is a new cc.
                ccs[nextcc] = [node]
                nextcc += 1
            else:
                # the node joins all the ccs in l.
                cc = ccs[l[0]]
                for j in l[1:]:
                    cc += ccs[j]
                    del ccs[j]
                cc += [node]

        self.ccs = list(ccs.values())
        #return len(self.ccs)
    
    def get_dot_graph(self, nodes=None, w_min=1):
        if nodes == None:
            nodes = self.nodes_list()
        g = pydot.Dot()
        g.set_type('graph')
        for i in range(1, len(nodes)):
            node = nodes[i]
            for j in range(i):
                # maybe:
                #if self.edge(str(node), str(nodes[j])):
                if self.edge_weight(node, nodes[j]) >= w_min:
                    e = pydot.Edge(node, nodes[j])
                    g.add_edge(e)
        return g
    
    def draw_graph(self, filename, nodes=None, w_min=1):
        """Draw the graph in a JPG file.
        """
        g = self.get_dot_graph(nodes, w_min=w_min)
        g.write_jpeg(filename, prog='dot')
