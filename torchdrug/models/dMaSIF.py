import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from math import sqrt
from torchdrug import core, layers
from torchdrug.core import Registry as R
from torchdrug.models.dMaSIF_utils import (atoms_to_points_normals, get_atom_features,
                                           diagonal_ranges, tangent_vectors, knn_atoms)
from pykeops.torch import LazyTensor

@R.register("models.GeoBind")
class GeoBind(nn.Module, core.Configurable):
    """
    Class for training Model weights
    """
    def __init__(self, input_dim, embedding_dim=16, output_dim=8, num_layers=4, radius=12.0, threshold=0.6):
        super(GeoBind, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.radius = radius
        self.threshold = threshold

        # ========== Atom Feature Weight ========== #
        self.feature_score = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.output_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.output_dim, 1),
            nn.Sigmoid()
        )

        # ========== Conv Layers ========== #
        self.conv_layers = DMaSIFSegment(
            input_dim=self.input_dim + 1,
            output_dim=self.embedding_dim,
            num_layers=self.num_layers,
            radius=self.radius
        )

        self.norm_out = nn.BatchNorm1d(self.embedding_dim)

        # ========== Prediction Layers ========== #
        self.out_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.output_dim, self.output_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.output_dim, 1), nn.Sigmoid()
        )

    def forward(self, graph, input, all_loss=None, metric=None):
        # ========== Atom Coord Transforming into Local Reference Frame ========== #
        points, normals, batch = self.local_frame_generate(graph, input)
        # ========== Atom Feature Embedding into Point Feature ========== #
        point_features, atom2point = self.point_embedding_encode(graph, input, points, batch)  # [N_points, 21]
        # ========== Point Embedding from Nearest Atom Feature ========== #
        weights = self.feature_score(point_features)
        nuv, ranges = self.local_frame_update(points, normals, weights, batch)  # update the nuv
        # ========== Atom Feature Embedding into Point Feature ========== #
        point_features = self.conv_layers(point_features, points, nuv, ranges)  # [N_points, 16]
        # ========== Atom Feature Embedding into Point Feature ========== #
        point_features = self.norm_out(point_features)
        pred = self.out_layers(point_features).squeeze(dim=-1)  # [N_points, 1]
        idx, dists = knn_atoms(graph.node_position, points, graph.atom2graph, batch, k=8)
        pred = pred[idx].mean(dim=-1)  # [N_atoms, 1]
        pocket_index = torch.nonzero(pred > self.threshold).squeeze()
        point_features = point_features[idx].mean(dim=-1)  # surface point feature for each residue
        torch.cuda.empty_cache()
        return pocket_index, point_features


    def local_frame_generate(self, graph, input):
        """
        :param graph: the packed grph
        :param input: the node feature
        :return: points, normals, batch
        """
        # ========== Protein Surface Points Sampling ========== #
        points, normals, batch = atoms_to_points_normals(
            atom_coords=graph.node_position, batch=graph.atom2graph, atomtypes=input
        )
        return points, normals, batch

    def local_frame_update(self, point_coords, normals, weights, batch):
        """
        Input arguments:
        - xyz, a point cloud encoded as an (N, 3) Tensor.
        - triangles, a connectivity matrix encoded as an (N, 3) integer tensor.
        - weights, importance weights for the orientation estimation, encoded as an (N, 1) Tensor.
        - radius, the scale used to estimate the local normals.
        - a batch vector, following PyTorch_Geometric's conventions.

        The routine updates the model attributes:
        - points, i.e. the point cloud itself,
        - nuv, a local oriented basis in R^3 for every point,
        - ranges, custom KeOps syntax to implement batch processing.
        """
        # 1. Save the vertices for later use in the convolutions ---------------
        ranges = diagonal_ranges(batch)  # KeOps support for heterogeneous batch processing need on CUDA
        points = point_coords / self.radius
        tangent_bases = tangent_vectors(normals)  # Tangent basis (N, 2, 3)
        # 3. Steer the tangent bases according to the gradient of "weights" ----
        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = LazyTensor(weights.view(1, -1, 1))  # (1, N, 1)
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # Normals:
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # Tangent basis:
        uv_i = LazyTensor(tangent_bases.view(-1, 1, 6))  # (N, 1, 6)
        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-rho2_ij).exp()  # (N, N, 1)
        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = window_ij * weights_j  # (N, N, 1) （1, N, 1）-> (N,N)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)
        # Support for heterogeneous batch processing:
        orientation_vector_ij.ranges = ranges  # Block-diagonal sparsity mask
        orientation_vector_i = orientation_vector_ij.sum(dim=1)  # (N, 2)
        orientation_vector_i = (orientation_vector_i + 1e-5)  # Just in case someone's alone...
        # 3.e) Normalize stuff:
        orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  # (N, 2)
        ex_i, ey_i = (orientation_vector_i[:, 0][:, None], orientation_vector_i[:, 1][:, None])  # (N,1)
        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (N, 2, 3)
        u_i, v_i = uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
        tangent_bases = torch.cat((ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=1).contiguous()  # (N, 6)
        # 4. Store the local 3D frame as an attribute --------------------------
        nuv = torch.cat((normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)), dim=1)
        return nuv, ranges

    def point_embedding_encode(self, graph, input, points, batch):
        point_features, atom2point = get_atom_features(
            point_coords=points, atom_coords=graph.node_position, point_batch=batch,
            atom_batch=graph.atom2graph, atom_feature=input, k=16
        )   # [N_points, k, 18]
        # mean/max the nearby residue feature
        return point_features.mean(dim=1), atom2point  # [N_points, 18]

# Original from dMaSIF.
class DMaSIFSegment(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, radius=12.0):
        super(DMaSIFSegment, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius

        self.conv_layers = nn.ModuleList(
            [DMaSIFConv(input_dim, output_dim, output_dim, radius)] +
            [DMaSIFConv(output_dim, output_dim, output_dim, radius) for i in range(num_layers - 1)]
        )

        self.output_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim)
            ) for i in range(num_layers)]
        )

        self.input_layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim)] +
            [nn.Linear(output_dim, output_dim) for i in range(num_layers - 1)]
        )

    def forward(self, features, points, nuv, ranges):
        layer_input = features
        for index, layer in enumerate(self.conv_layers):
            hidden = layer(points, nuv, layer_input, ranges)
            hidden = self.output_layers[index](hidden)
            layer_input = self.input_layers[index](layer_input)
            layer_input = layer_input + hidden
        return layer_input


class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases. 
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class DMaSIFConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, radius=12.0):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features
        O = out_channels is the dimension of the output features
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)

        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:
          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)
          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 1 linear layer if "cheap" is True;
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.
          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)
        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.
        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
            cheap (bool, optional): shall we use a 1-layer deep Filter,
                instead of a 2-layer deep MLP? Defaults to False.
        """

        super(DMaSIFConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.Radius = radius

        self.Cuts = 8  # Number of hidden units for the 3D MLP Filter.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "hidden_dim" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if self.hidden_dim < self.heads_dim:
            self.heads_dim = self.hidden_dim

        if self.hidden_dim % self.heads_dim != 0:
            raise ValueError(f"The dimension of the hidden units ({self.hidden_dim})" \
                             + f"should be a multiple of the heads dimension ({self.heads_dim}).")
        else:
            self.num_heads = self.hidden_dim // self.heads_dim

        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # (H, H) + (H,)
            nn.LeakyReLU(negative_slope=0.2), )  # (H,)
        self.norm_in = nn.GroupNorm(4, self.hidden_dim)

        # 3D convolution filters, encoded as an MLP:
        self.conv = nn.Sequential(
            nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
            nn.ReLU(),  # KeOps does not support well LeakyReLu
            nn.Linear(self.Cuts, self.hidden_dim),
        )  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.output_dim, self.output_dim),  # (O, O) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
        )  # (O,)

        self.norm_out = nn.GroupNorm(4, self.output_dim)

        # Custom initialization for the MLP convolution filters:
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()
            nn.init.uniform_(
                self.conv[2].weight,
                a=-1 / np.sqrt(self.Cuts),
                b=1 / np.sqrt(self.Cuts),
            )
            nn.init.normal_(self.conv[2].bias)
            self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()

    def forward(self, points, nuv, features, ranges):
        """
        Performs a quasi-geodesic interaction step.
        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)
        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.
        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): low-level format
                to support batch processing, as described in the KeOps documentation.
                In practice, this will be built by a higher-level object
                to encode the relevant "batch vectors" in a way that is convenient
                for the KeOps CUDA engine. Defaults to None.
        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # 1. Transform the input features: -------------------------------------
        features = self.net_in(features)  # (N, I) -> (N, H)
        features = features.transpose(1, 0)[None, :, :]  # (1,H,N)
        features = self.norm_in(features)
        features = features[0].transpose(1, 0).contiguous()  # (1, H, N) -> (N, H)
        # 2. Compute the local "shape contexts": -------------------------------
        # 2.a Normalize the kernel radius:
        points = points / (sqrt(2.0) * self.Radius)  # (N, 3)
        # 2.b Encode the variables as KeOps LazyTensors
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # WARNING - Here, we assume that the normals are fixed:
        normals = (
            nuv[:, 0, :].contiguous().detach()
        )  # (N, 3) - remove the .detach() if needed
        # Local bases:
        nuv_i = LazyTensor(nuv.view(-1, 1, 9))  # (N, 1, 9)
        # Normals:
        n_i = nuv_i[:3]  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # To avoid register spilling when using large embeddings, we perform our KeOps reduction
        # over the vector of length "self.Hidden = self.num_heads * self.heads_dim"
        # as self.num_heads reduction over vectors of length self.heads_dim (= "Hd" in the comments).
        head_out_features = []
        for head in range(self.num_heads):
            # Extract a slice of width Hd from the feature array
            head_start = head * self.heads_dim
            head_end = head_start + self.heads_dim
            head_features = features[:, head_start:head_end].contiguous()  # (N, H) -> (N, Hd)
            # Features:
            f_j = LazyTensor(head_features[None, :, :])  # (1, N, Hd)
            # Convolution parameters:
            A_1, B_1 = self.conv[0].weight, self.conv[0].bias  # (C, 3), (C,)
            # Extract a slice of Hd lines: (H, C) -> (Hd, C)
            A_2 = self.conv[2].weight[head_start:head_end, :].contiguous()
            # Extract a slice of Hd coefficients: (H,) -> (Hd,)
            B_2 = self.conv[2].bias[head_start:head_end].contiguous()
            a_1 = LazyTensor(A_1.view(1, 1, -1))  # (1, 1, C*3)
            b_1 = LazyTensor(B_1.view(1, 1, -1))  # (1, 1, C)
            a_2 = LazyTensor(A_2.view(1, 1, -1))  # (1, 1, Hd*C)
            b_2 = LazyTensor(B_2.view(1, 1, -1))  # (1, 1, Hd)
            # 2.c Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij = (-d2_ij).exp()  # (N, N, 1)
            # 2.d Local MLP:
            # Local coordinates:
            X_ij = nuv_i.matvecmult(x_j - x_i)  # (N, N, 3)
            # MLP:
            X_ij = a_1.matvecmult(X_ij) + b_1  # (N, N, C)
            X_ij = X_ij.relu()  # (N, N, C)
            X_ij = a_2.matvecmult(X_ij) + b_2  # (N, N, Hd)
            X_ij = X_ij.relu()
            # 2.e Actual computation:
            F_ij = window_ij * X_ij * f_j  # (N, N, Hd)
            F_ij.ranges = ranges  # Support for batches and/or block-sparsity
            head_out_features.append(ContiguousBackward().apply(F_ij.sum(dim=1)))  # (N, Hd)

        # Concatenate the result of our num_heads "attention heads":
        features = torch.cat(head_out_features, dim=1)  # num_heads * (N, Hd) -> (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.transpose(1, 0)[None, :, :]  # (1,O,N)
        features = self.norm_out(features)
        features = features[0].transpose(1, 0).contiguous()

        return features

