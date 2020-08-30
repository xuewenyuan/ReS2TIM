import sys

class ShortestDis():
 
    def __init__(self, graph):
        self.V = len(graph)
        #self.graph = [[0 for column in range(len(vertices))] 
        #              for row in range(len(vertices))]
        self.graph = graph
 
    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node,"\t",dist[node])
 
    def minDistance(self, dist, sptSet):
 
        _min = sys.maxsize

        min_index = -1
 
        for v in range(self.V):
            if dist[v] < _min and sptSet[v] == False:
                _min = dist[v]
                min_index = v
 
        return min_index
    
    def cpmin(self, src, goal):
        dist, paths = self.dijkstra(src)
        _min = sys.maxsize
        minpath = []
        for x in range(len(dist)):
            if x in goal and dist[x]<_min:
                _min = dist[x]
                minpath = paths[x]
        #print(_min)
        return minpath
    
    def dijkstra(self, src):
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        paths = [[i] for i in range(self.V)]
        for cout in range(self.V):
 
            u = self.minDistance(dist, sptSet)
 
            sptSet[u] = True
 
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                   dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
                        paths[v] = paths[u] + paths[v]
 
        #self.printSolution(dist)
        #print paths
        return dist, paths

if __name__ == '__main__':
    
    graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
             [4, 0, 8, 0, 0, 0, 0, 11, 0],
             [0, 8, 0, 7, 0, 4, 0, 0, 2],
             [0, 0, 7, 0, 9, 14, 0, 0, 0],
             [0, 0, 0, 9, 0, 10, 0, 0, 0],
             [0, 0, 4, 14, 10, 0, 2, 0, 0],
             [0, 0, 0, 0, 0, 2, 0, 1, 6],
             [8, 11, 0, 0, 0, 0, 1, 0, 7],
             [0, 0, 2, 0, 0, 0, 6, 7, 0]
            ]
    g = ShortestDis(graph)
    g.cpmin(0, [1,3,7])