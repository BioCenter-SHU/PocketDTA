import torch
from torch import nn
from torch.nn import functional as F
import math
from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinBERT")
class ProteinBERT(nn.Module, core.Configurable):
    """
    Protein BERT proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int, optional): hidden dimension
        num_layers (int, optional): number of Transformer blocks
        num_heads (int, optional): number of attention heads
        intermediate_dim (int, optional): intermediate hidden dimension of Transformer block
        activation (str or function, optional): activation function
        hidden_dropout (float, optional): dropout ratio of hidden features
        attention_dropout (float, optional): dropout ratio of attention maps
        max_position (int, optional): maximum number of positions
    """

    def __init__(self, input_dim, hidden_dim=768, num_layers=12, num_heads=12, intermediate_dim=3072,
                 activation="gelu", hidden_dropout=0.1, attention_dropout=0.1, max_position=8192):
        super(ProteinBERT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position = max_position

        self.num_residue_type = input_dim
        self.embedding = nn.Embedding(input_dim + 3, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(layers.ProteinBERTBlock(hidden_dim, intermediate_dim, num_heads,
                                                       attention_dropout, hidden_dropout, activation))
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)

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
        input = graph.residue_type
        size_ext = graph.num_residues
        # Prepend BOS
        bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.num_residue_type
        input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        # Append EOS
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * (self.num_residue_type + 1)
        input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        # Padding
        input, mask = functional.variadic_to_padded(input, size_ext, value=self.num_residue_type + 2)
        mask = mask.long().unsqueeze(-1)

        input = self.embedding(input)
        position_indices = torch.arange(input.shape[1], device=input.device)
        input = input + self.position_embedding(position_indices).unsqueeze(0)
        input = self.layer_norm(input)
        input = self.dropout(input)

        for layer in self.layers:
            input = layer(input, mask)

        residue_feature = functional.padded_to_variadic(input, graph.num_residues)

        graph_feature = input[:, 0]
        graph_feature = self.linear(graph_feature)
        graph_feature = F.tanh(graph_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }


class SelfAttention(nn.Module):
    """
    TransformerCPI Decoder model for DTA prediction.
    """
    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
 
        # query = key = value shape: [batch_size, sent len, hidden_dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch_size, sentence_len, hidden_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # K, V = [batch_size, num_heads, sentence_len_K, head_size]
        # Q = [batch_size, num_heads, sentence_len_Q, head_size]
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_size)
        # scores = [batch_size, num_heads, sentence_len_Q, sentence_len_K]
        # TODO: mask shape error
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        # 补全位置上的值成为一个非常大的负数（可以是负无穷），这样的话，经过Softmax层的时候，这些位置上的概率就是0
        attention = self.dropout(F.softmax(scores, dim=-1))
        # attention = [batch_size, num_heads, sentence_len_Q, sentence_len_K]

        atten_output = torch.matmul(attention, V)
        # atten_output = [batch_size, num_heads, sentence_len_Q, head_size]

        atten_output = atten_output.permute(0, 2, 1, 3).contiguous()
        # atten_output = [batch_size, sentence_len_Q, num_heads, head_size]

        atten_output = atten_output.view(batch_size, -1, self.hidden_dim)
        # atten_output = [batch_size, sentence_len_Q, hidden_dim]

        atten_output = self.fc(atten_output)
        # atten_output = [batch_size, sentence_len_Q, hidden_dim]

        return atten_output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim  # 64
        self.pf_dim = pf_dim  # 256
        self.linear_1 = nn.Conv1d(hidden_dim, pf_dim, 1)  
        self.linear_2 = nn.Conv1d(pf_dim, hidden_dim, 1)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.dropout(F.relu(self.linear_1(x)))
        # x = [batch size, pf dim, sent len]  64->256

        x = self.linear_2(x)
        # x = [batch size, hid dim, sent len]  256->64

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, pf_dim, dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.self_atten = SelfAttention(hidden_dim, num_heads, dropout)
        self.self_atten = SelfAttention(hidden_dim, num_heads, dropout)
        self.position_feed = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_embed, target_embed, drug_mask=None, target_mask=None):
        """ 
        drug: [batch_size, compound_len, atom_dim]
        target: [batch_size, protein_len, hidden_dim]
        drug_mask: [batch_size, compound_len, 1]
        target_mask: [batch_size, protein_len, 1]
        """
        drug_embed = self.layernorm(drug_embed + self.dropout(self.self_atten(drug_embed, drug_embed, drug_embed, drug_mask)))
        drug_embed = self.layernorm(drug_embed + self.dropout(self.self_atten(drug_embed, target_embed, target_embed, target_mask)))
        output = self.layernorm(drug_embed + self.dropout(self.position_feed(drug_embed)))
        return output


@R.register("models.DTAPredictor")
class DTAPredictor(nn.Module, core.Configurable):
    """ compound feature extraction."""
    def __init__(self, target_dim, drug_dim, hidden_dim, num_layers=3, num_heads=8, pf_dim=256, dropout=0.1):
        super(DTAPredictor, self).__init__()
        self.output_dim = drug_dim
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers
        self.num_heads = num_heads 
        self.pf_dim = pf_dim  # for position conv1d hidden_dim

        self.decoder_layer = DecoderLayer(hidden_dim, num_heads, pf_dim, dropout)  
        self.layers = nn.ModuleList([self.decoder_layer for _ in range(num_layers)])

        self.drug_linear = nn.Linear(drug_dim, hidden_dim)
        self.target_linear = nn.Linear(target_dim, hidden_dim)

        self.drug_readout = layers.AttentionReadout(input_dim=hidden_dim, type='node')
        self.target_readout = layers.AttentionReadout(input_dim=hidden_dim, type='residue')
        self.mlp = layers.MLP(hidden_dim * 2 , [hidden_dim * 2, hidden_dim, 1], batch_norm=True)

    def forward(self, target_embed, drug_embed, target_graph, drug_graph):
        """ 
        Input:
            target_embed shape: [batch_size, num_residues, target_dim]
            drug_embed shape: [all_atoms, drug_dim]
        """
        # target_embed shape: [all_residues, target_dim] -> [batch_size, max_num_residues, target_dim]
        # drug_embed shape: [all_atoms, drug_dim] -> [batch_size, max_num_atoms, drug_dim]
        target_embed = layers.functional.variadic_to_padded(target_embed, target_graph.num_residues)[0]
        drug_embed = layers.functional.variadic_to_padded(drug_embed, drug_graph.num_nodes)[0]
        
        # get the padding mask for transformers blocks
        target_mask, drug_mask = self.make_masks(target_graph.num_residues, drug_graph.num_nodes, 
                                                 target_embed.shape[1], drug_embed.shape[1])

        # project target and drug into the space
        # target_embed shape: [batch_size, max_num_residues, target_dim] -> [batch_size, max_num_residues, hidden_dim]
        # drug_embed shape: [batch_size, max_num_atoms, drug_dim] -> [batch_size, max_num_atoms, hidden_dim]
        target_embed = self.target_linear(target_embed)
        drug_embed = self.drug_linear(drug_embed)

        for layer in self.layers:
            drug_embed = layer(drug_embed, target_embed, drug_mask, target_mask)

        # Use norm to determine which atom is significant
        # original method!
        # norm = torch.norm(hidden_embed, dim=2)
        # norm = F.softmax(norm, dim=1)
        # sum = torch.zeros((drug_embed.shape[0], self.hidden_dim)).to(self.device)
        # for i in range(norm.shape[0]):
        #     for j in range(norm.shape[1]):
        #         v = drug_embed[i, j, ]
        #         v = v * norm[i, j]
        #         sum[i, ] += v
        # sum = [batch_size, hidden_dim]
        # TODO：这里可以修改作为创新点，或者将PSSA引入
        # sum = torch.sum(hidden_embed * norm[:, :, None], axis=1)
        # original method!
        # TODO：这里可以修改作为创新点，或者将PSSA引入
        target_embed = layers.functional.padded_to_variadic(target_embed, target_graph.num_residues)
        target_embed = self.target_readout(target_graph, target_embed)
        drug_embed = layers.functional.padded_to_variadic(drug_embed, drug_graph.num_nodes)
        drug_embed = self.drug_readout(drug_graph, drug_embed)
        pred = self.mlp(torch.cat([target_embed, drug_embed], dim=-1))

        return pred


    def make_masks(self, protein_num_list, atom_num_list, protein_max_len, compound_max_len):
        protein_axes = torch.arange(0, protein_max_len, device=self.device).view(1, -1)
        protein_mask = (protein_axes < protein_num_list.view(-1, 1)).unsqueeze(1).unsqueeze(2)
        compound_axes = torch.arange(0, compound_max_len, device=self.device).view(1, -1)
        compound_mask = (compound_axes < atom_num_list.view(-1, 1)).unsqueeze(1).unsqueeze(3)
        return protein_mask, compound_mask


