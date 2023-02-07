"""
    def cluster(self, n_clusters):
        distanceMatrix = self.computeDistance()
        clusters = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average').fit(distanceMatrix)
        return clusters
    
    # Partner Metric
    def computeDistance(self):
        def baseline(x, y):
            partner_0 = sqrt((x[0] - x[::-1][0]) ** 2 + (x[1] - x[::-1][1]) ** 2)
            partner_1 = sqrt((y[0] - y[::-1][0]) ** 2 + (y[1] - y[::-1][1]) ** 2)
            distance = sqrt((partner_0 - partner_1) ** 2)
            return distance
        
        distanceMatrix = np.asarray([[baseline(state_0, state_1) for state_1 in self.states] for state_0 in self.states])
        return distanceMatrix
    
    Corner Metric
    def computeDistance(self, states):
        distanceMetric = lambda x, y: sqrt((x - y) ** 2)
        cornerMetric = lambda x, corner: sqrt((x[0] - corner[0]) ** 2 + (x[1] - corner[1]) ** 2)
        corners = [[0,0], [self.grid_dims[0] - 1, 0], [0, self.grid_dims[1] - 1], [self.grid_dims[0] - 1, self.grid_dims[1] - 1]]
        
        cornerMatrix = [min(cornerMetric(state, corner) for corner in corners) for state in states]
        distanceMatrix = np.asarray([[distanceMetric(state_0, state_1) for state_1 in cornerMatrix] for state_0 in cornerMatrix])
        return distanceMatrix
    
    Euclidean Reciprocal Metric
    def computeDistance(self, states):
        distanceMetric = lambda x, y: np.reciprocal(sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)) if x != y else 0
        distanceMatrix = np.asarray([[distanceMetric(state_0, state_1) for state_1 in states] for state_0 in states])
        return distanceMatrix
    
    def communicate(self, paths, clusters, num_communicators):
        messages = []
        for i in range(num_communicators):
            messages.append(clusters.predict(paths[i]).tolist())   
        return messages
    """
