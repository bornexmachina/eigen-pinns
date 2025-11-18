import yaml


class PINNConfig:
    def __init__(self, mesh_file, n_modes, hierarchy, k_neighbors, epochs, learning_rate, corrector_scale,
                 weight_residual, weight_orthogonal, weight_projection, gradient_clipping, weight_decay, log_every,
                 hidden_layers, dropout, normalization_eps, prolongation_neighbors, knn_graph_neighbors, verbose):
        self.mesh_file = mesh_file
        self.n_modes = n_modes
        self.hierarchy = hierarchy
        self.k_neighbors = k_neighbors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.corrector_scale = corrector_scale
        self.weight_residual = weight_residual
        self.weight_orthogonal = weight_orthogonal
        self.weight_projection = weight_projection
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.normalization_eps = normalization_eps
        self.prolongation_neighbors = prolongation_neighbors
        self.knn_graph_neighbors = knn_graph_neighbors
        self.verbose = verbose

    @classmethod
    def from_yaml(cls, file):
        with open(file, "r") as f:
            parameters = yaml.safe_load(f)

        return cls(**parameters)