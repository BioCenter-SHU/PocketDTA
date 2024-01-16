from collections.abc import Sequence
from math import pi as PI
import torch
from torch import nn
from torch_scatter import scatter_min, scatter_add
from torch.nn import functional as F
from torchdrug import core, layers
from torchdrug.core import Registry as R
from torchdrug.layers import MessagePassingBase

from .comenet_features import angle_emb, torsion_emb

@R.register("models.ComENet")
class ComENet(nn.Module, core.Configurable):
    """
    ComENet from `ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs`.
    https://doi.org/10.48550/arXiv.2206.08515

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions for interaction layers
        middle_dims (list of int): middle dimensions for self-atom layers
        cutoff (float, optional): cutoff distance for interatomic interactions
        num_radial (int, optional): number of radial basis functions
        num_spherical (int, optional): number of spherical harmonics
        num_mlp_layer (int, optional): number of MLP layers
    """

    def __init__(self, input_dim, hidden_dims, middle_dim=64, edge_input_dim=None, cutoff=8.0, num_radial=3, num_spherical=2, 
                    num_mlp_layer=3, short_cut=True, batch_norm=True, activation="silu", concat_hidden=False, readout="sum"):
        super(ComENet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.hidden_dim = hidden_dims[0]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(ComEConv(self.dims[i], self.dims[i + 1], self.hidden_dim, middle_dim, edge_input_dim, layer_hidden_dims,  
                                        cutoff, num_radial, num_spherical, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node/edge/graph representation(s).

        Require the graph(s) to have node attribute ``node_position``.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
    

class ComEConv(MessagePassingBase):
    """
    ComENet from `ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs`.
    https://doi.org/10.48550/arXiv.2206.08515
    
    The Interaction Blocks in the paper.

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        hidden_dim (int, optional): hidden dimension
        hidden_dim (int, optional): hidden dimension
        cutoff (float, optional): maximal scale for RBF kernels
        num_gaussian (int, optional): number of RBF kernels
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, hidden_dim, middle_dim, edge_input_dim, hidden_dims=None, cutoff=8.0,
                num_radial=3, num_spherical=2, batch_norm=True, activation="silu"):
        super(ComEConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim
        self.cutoff = cutoff
        
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.local_layer = torsion_emb(num_radial, num_spherical, cutoff)
        self.global_layer = angle_emb(num_radial, num_spherical, cutoff)
        self.local_linear = layers.MLP(num_radial * num_spherical ** 2, [middle_dim, hidden_dim], activation)
        self.global_linear = layers.MLP(num_radial * num_spherical, [middle_dim, hidden_dim], activation)
        self.down_project = layers.MLP(hidden_dim * 2, [self.hidden_dim], activation)
        
        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(hidden_dim, list(hidden_dims) + [output_dim], activation)  # input_dim
        if edge_input_dim:                           
            self.edge_linear = nn.Linear(edge_input_dim, hidden_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        """
        The graph should be a radius_graph after a GraphConstruction.
        The graph.edge_feature should be the edge length
        """
        # add self loop
        # node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_in = graph.edge_list[:, 0]
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        distance = graph.edge_feature.squeeze() 
        edge_feature = self.edge_layer(graph, distance, self.cutoff)
        # Node(node_in atom_feature) embedding
        node_embedding = self.input_layer(input)
        node_message = node_embedding[node_in] 
        message = edge_feature + node_message + graph.edge_feature
        message /= (degree_in[node_in].sqrt() + 1e-10)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def combine(self, input, update):
        output = self.mlp(update)  # error place
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    
    def edge_layer(self, graph, distance, cutoff):
        # Angle embedding tensor([theta, phi, tau])
        angle_feature = self.angle_layer(graph, distance, cutoff)
        local_feature = self.local_layer(distance, angle_feature[:,0], angle_feature[:,1])
        local_feature = self.local_linear(local_feature)
        global_feature = self.global_layer(distance, angle_feature[:,2])
        global_feature = self.global_linear(global_feature)
        # Concate three embedding as the message for the edge 
        edge_feature = self.down_project(torch.concat([local_feature, global_feature], dim=1))
        return edge_feature

    def angle_layer(self, graph, distance, cutoff):
        """
        Paper: 
            Formally, for a center node i, we let N_i and N^2_i denote two sets of indices of i's 1hop 
            and 2-hop neighboring nodes, respectively. We also define any node i and its 1-hop neighborhood 
            as a local structure(one edge from i->N_i). Then the whole 2-hop neighborhood of node i can be viewed as 1 + |Ni| 
            local structures centered in i and Ni(two edges from current i and one-hop node j, i->j->N^2_i), 
            defined as L_i and L_ij, j∈N_i , respectively.
        Spherical Coordinate System:
            For each center node i, all the other neighborhood node can be noted as (d, θ, φ) to calculate the relative position.
            Then use the angle info to aggregate the edge and node representation into the center node i.
        Steps:
            1.building local coordinate systems: 
                First, we build a light local coordinate system for any node i's corresponding local structure.
                Similarly, the center node i serves as the origin.
            2.defining z-axis:
                Then z-axis is defined as the direction from i to its nearest neighbor fi, and xz-plane is further formed by 
                z-axis and i's second nearest neighbor si.
            3.picking reference nodes:
                using a algorithm to get i's nearest neighbor fi, and i's second nearest neighbor si.
            4.computing (d, θ, φ):
                Finally, the tuple (d, θ, φ) is computed within 1-hop neighborhood with a complexity of O(nk).
        Return:
            torch.stack([theta, phi, tau], dim=1)
        """
        # source node j(node_in), target node i(node_out)
        node_in, node_out = graph.edge_list.t()[:2] 
        vectors = graph.node_position[node_out] - graph.node_position[node_in]
        # Get the nearest neighbor f_i
        _, source_index_f = scatter_min(distance, node_out, dim_size=graph.num_node) 
        source_index_f[source_index_f >= len(node_out)] = 0 
        source_node_f = node_in[source_index_f]  # nearest node id
        # To exclude the nearest node in second nearest selecting, add a publishment
        source_punishment = torch.zeros_like(distance).to(distance.device)  
        source_punishment[source_index_f] = cutoff
        source_punished_distance = distance + source_punishment
        
        # Get the second nearest s_i
        _, source_index_s = scatter_min(source_punished_distance, node_out, dim_size=graph.num_node)
        source_index_s[source_index_s >= len(node_out)] = 0 
        source_node_s = node_in[source_index_s]

        # The same for the target node
        _, target_index_f = scatter_min(distance, node_in, dim_size=graph.num_node)
        target_index_f[target_index_f >= len(node_in)] = 0 
        target_node_f = node_out[target_index_f]
        # To exclude the nearest node in second nearest selecting, add a publishment
        target_punishment = torch.zeros_like(distance).to(distance.device)
        target_punishment[target_index_f] = cutoff
        target_punishment_distance = distance + target_punishment
        # Get the second nearest s_i
        _, target_index_s = scatter_min(target_punishment_distance, node_in, dim_size=graph.num_node)
        target_index_s[target_index_s >= len(node_in)] = 0 
        target_node_s = node_out[target_index_s]

        source_node_f = source_node_f[node_out]
        source_node_s = source_node_s[node_out]

        target_node_f = target_node_f[node_in]
        target_node_s = target_node_s[node_in]

        # tau: (iref, node_out, node_in, jref)
        # when compute tau, do not use source_node_f, target_node_f as ref for i and j,
        # because if source_node_f = j, or target_node_f = i, the computed tau is zero
        # so if source_node_f = j, we choose iref = source_node_s
        # if target_node_f = i, we choose jref = target_node_s
        mask_iref = source_node_f == node_in
        iref = torch.clone(source_node_f)
        iref[mask_iref] = source_node_s[mask_iref]
        idx_iref = source_index_f[node_out]
        idx_iref[mask_iref] = source_index_s[node_out][mask_iref]

        mask_jref = target_node_f == node_out
        jref = torch.clone(target_node_f)
        jref[mask_jref] = target_node_s[mask_jref]
        idx_jref = target_index_f[node_in]
        idx_jref[mask_jref] = target_index_s[node_in][mask_jref]

        pos_ji, pos_if, pos_is, pos_iref, pos_jref = (
            vectors,
            vectors[source_index_f][node_out],
            vectors[target_index_f][node_out],
            vectors[idx_iref],
            vectors[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_if).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_if).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + PI

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_if)
        plane2 = torch.cross(-pos_ji, pos_is)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + PI

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + PI

        return torch.stack([theta, phi, tau], dim=1)
