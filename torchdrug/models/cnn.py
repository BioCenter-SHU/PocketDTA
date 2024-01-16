from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinResNet")
class ProteinResNet(nn.Module, core.Configurable):
    """
    Protein ResNet proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size=3, stride=1, padding=1,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=False,
                 dropout=0, readout="attention"):
        super(ProteinResNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.position_embedding = layers.SinusoidalPositionEmbedding(hidden_dims[0])
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[0])
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.ProteinResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "attention":
            self.readout = layers.AttentionReadout(self.output_dim, "residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_feature.float()
        input, mask = functional.variadic_to_padded(input, graph.num_residues, value=self.padding_id)
        mask = mask.unsqueeze(-1)

        input = self.embedding(input) + self.position_embedding(input).unsqueeze(0)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        input = input * mask
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input, mask)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        residue_feature = functional.padded_to_variadic(hidden, graph.num_residues)
        graph_feature = self.readout(graph, residue_feature)
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }


@R.register("models.ProteinConvolutionalNetwork")
class ProteinConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Protein Shallow CNN proposed in `Is Transfer Learning Necessary for Protein Landscape Prediction?`_.

    .. _Is Transfer Learning Necessary for Protein Landscape Prediction?:
        https://arxiv.org/pdf/2011.03443.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean``, ``max`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, kernel_size=3, stride=1, padding=1,
                activation='relu', short_cut=False, concat_hidden=False, readout="max"):
        super(ProteinConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # self.embed = nn.Embedding(21,  128, padding_idx=0)  # TODO: added for sequence embedding
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "max":
            self.readout = layers.MaxReadout("residue")
        elif readout == "attention":
            self.readout = layers.AttentionReadout(self.output_dim, "residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations shape (all_residues, 21)
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        input = graph.residue_feature.float() # original method (19862, 21)!
        # input = graph.residue_feature.nonzero()[:, 1]  # input shape (all_residues,)as the residue index for all residues
        # input = self.embed(input).float()  # convert the index to lookup a embedding shape (all_residues, embedding_dim)
        input = functional.variadic_to_padded(input, graph.num_residues, value=self.padding_id)[0]  # shape (32, 1000, 64)
        
        hiddens = []
        layer_input = input

        # Conv1D input shape (N, C, L)
        for layer in self.layers:
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input  # * 0.5  # added for short cut
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        residue_feature = functional.padded_to_variadic(hidden, graph.num_residues)
        graph_feature = self.readout(graph, residue_feature)
        
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }


@R.register("models.ConvolutionalNeuralNetwork")
class FPConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Reference:
    Conv: 
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
                        dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    MaxPooling: 
        torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, 
                        return_indices=False, ceil_mode=False)
    Parameters:
        input_dim (int): input dimension, drug is 21 for now and protein is 67 for now
        hidden_dims (list of int): filters dim of each conv layer, which is also the input and output dim
        kernel_size (list of int): size of convolutional kernel in each layer
        stride (list of int): stride of convolution in each layer
        padding (list of int): padding added to both sides of the input in each layer
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, hidden_dims, kernel_size=3, stride=1, padding=1, 
                 activation='relu', short_cut=False, concat_hidden=False):
        super(FPConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )


    def forward(self, graph):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            input (Tensor): input molecule graph with grpha representations(fingerprint)
        """
        input = graph.graph_feature.float().unsqueeze(dim=1)
 
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        return hidden.squeeze(dim=1)


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)


@R.register("models.StackCNN")
class StackCNN(nn.Module, core.Configurable):
    """
    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        num_layer (int): the stacked layer numbers
        kernel_size (list of int): size of convolutional kernel in each layer
        stride (list of int): stride of convolution in each layer
        padding (list of int): padding added to both sides of the input in each layer
        activation (str or function, optional): activation function
        readout (str, optional): readout function
    """

    def __init__(self, layer_nums, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            Conv1dReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding)
        )
        if layer_nums > 1:
            for i in range(layer_nums-1):
                self.layers.append(
                    Conv1dReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
                )
        self.layers.append(nn.AdaptiveAvgPool1d(1))


    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).
        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        layer_input = input
        hiddens = []

        for layer in self.layers:
            hidden = layer(layer_input)
            hiddens.append(hidden)
            layer_input = hidden
        # graph_feature shape: [512, 96]
        graph_feature = hiddens[-1].squeeze(dim=-1)
        
        return graph_feature


@R.register("models.MCNN")
class DenseConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Convolutional Neural Network proposed in `MGraphDTA: Deep Multiscale Graph Neural Network`_.

    .. _MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction:

    Parameters:
        input_dim (int): input dimension
        block_num (int): block nums of different CNN
        embedding_dim (int): dims of the word embedding
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, block_num=3, embedding_dim=128):
        super(DenseConvolutionalNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 96
        self.padding_id = input_dim - 1

        self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim, padding_idx=0)

        self.layers = nn.ModuleList()
        for block_idx in range(block_num):
            self.layers.append(
                StackCNN(layer_nums=block_idx+1, in_channels=embedding_dim, out_channels=96, kernel_size=3)
            )

        self.linear = nn.Linear(block_num * 96, 96)


    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = graph.residue_feature.nonzero()[:, 1]  # layer_input shape: [382919, 21] -> [382919,] from one-hot to number sequence
        layer_input = self.embed(layer_input).float()  # layer_input shape: [382919, ] -> [382919, 128]  后续可以把训练好的权重存下来
        # layer_input shape: [382919, 128] -> [512, 1200, 128]-> [512, 128, 1200] for Conv1D shape require (N, C, L)
        layer_input = functional.variadic_to_padded(layer_input, graph.num_residues,
                                                    value=self.padding_id)[0].permute(0, 2, 1)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            hiddens.append(hidden)

        # graph_feature shape: [2045, 228] -> [2045, 96]
        graph_feature = self.linear(torch.cat(hiddens, dim=-1))

        return {
            "graph_feature": graph_feature
        }
