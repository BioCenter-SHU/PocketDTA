import math

import torch
import torch.nn as nn


def bias_dropout_add(x: torch.Tensor, bias: torch.Tensor, dropmask: torch.Tensor,
                     residual: torch.Tensor, prob: float) -> torch.Tensor:
    out = (x + bias) * F.dropout(dropmask, p=prob, training=False)
    out = residual + out
    return out


def bias_ele_dropout_residual(ab: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
                              dropout_mask: torch.Tensor, Z_raw: torch.Tensor,
                              prob: float) -> torch.Tensor:
    return Z_raw + F.dropout(dropout_mask, p=prob, training=True) * (g * (ab + b))


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


class Transition(nn.Module):

    def __init__(self, d, n=4):
        super(Transition, self).__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, n * d)
        self.linear2 = nn.Linear(n * d, d)

    def forward(self, src):
        x = self.norm(src)
        x = self.linear2(F.relu(self.linear1(x)))
        return src + x


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads):
        """
        Args:
            query_dim: Input dimension of query data
            key_dim: Input dimension of key data
            value_dim: Input dimension of value data
            hidden_dim: Per-head hidden dimension
            num_heads: Number of attention heads
            gating: Whether the output should be gated using query data
        """
        super(MultiHeadAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.linear_q = nn.Linear(query_dim, hidden_dim * num_heads, bias=False)
        self.linear_k = nn.Linear(key_dim, hidden_dim * num_heads, bias=False)
        self.linear_v = nn.Linear(value_dim, hidden_dim * num_heads, bias=False)
        self.linear_o = nn.Linear(hidden_dim * num_heads, query_dim)

        self.linear_g = nn.Linear(query_dim, hidden_dim * num_heads)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def prep_qkv(self, query_data, key_data):
        # [Batch_size, N_res, N_atom, num_heads * hidden_dim]
        q = self.linear_q(query_data)
        k = self.linear_k(key_data)
        v = self.linear_v(key_data)

        # [Batch_size, N_res, N_atom, num_heads, hidden_dim]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        q /= math.sqrt(self.hidden_dim)

        return q, k, v

    def wrap_up(self, output, query_data):

        g = self.sigmoid(self.linear_g(query_data))
        # [*, Q, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.num_heads, -1))
        output = output * g

        # [*, Q, H * C_hidden]
        output = flatten_final_dims(output, 2)
        # [*, Q, C_q]
        output = self.linear_o(output)

        return output

    def attention(self, query, key, value, biases):
        # [Batch_size, N_res, num_heads, N_atom, hidden_dim]
        query = permute_final_dims(query, (1, 0, 2))
        # [Batch_size, N_res, num_heads, hidden_dim, N_atom]
        key = permute_final_dims(key, (1, 2, 0))
        # [Batch_size, N_res, num_heads, N_atom, hidden_dim]
        value = permute_final_dims(value, (1, 0, 2))
        # [Batch_size, N_res, num_heads, N_atom, N_atom]
        scores = torch.matmul(query, key)
        if biases is not None:
            for b in biases:
                scores = scores + b

        attn_weights = self.softmax(scores)
        # [*, H, Q, C_hidden]
        attn_outpout = torch.matmul(attn_weights, value)
        # [*, Q, H, C_hidden]
        attn_outpout = attn_outpout.transpose(-2, -3)

        return attn_outpout

    def forward(self, query_data, key_data, biases=None):
        """
        Args:
            query_data: [*, Q, C_q] query data
            output: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        if (biases is None):
            biases = []

        q, k, v = self.prep_qkv(query_data, key_data)
        output = self.attention(q, k, v, biases)
        output = self.wrap_up(output, query_data)

        return output


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10. OPM can combine the pair info into interaction map.
    """

    def __init__(self, target_dim, drug_dim, interaction_dim=128, hidden_dim=32, eps=1e-3):
        """
        Args:
            target_dim: Protein embedding dimension
            drug_dim: Drug embedding dimension
            interaction_dim: Interaction embedding dimension
            hidden_dim: Hidden dimension for compact learning
        """
        super(OuterProductMean, self).__init__()

        self.target_dim = target_dim
        self.drug_dim = drug_dim
        self.interaction_dim = interaction_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.layernorm_target = nn.LayerNorm(target_dim)
        self.layernorm_drug = nn.LayerNorm(drug_dim)
        self.linear_target = nn.Linear(target_dim, hidden_dim)
        self.linear_drug = nn.Linear(drug_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim ** 2, interaction_dim)

    def opm(self, target_feature, drug_feature):
        # [Batch_size, N_res, 1, hidden_dim] * [Batch_size, N_atom, 1, hidden_dim] -> [Batch_size, N_res, N_atom, hidden_dim, hidden_dim]
        outer = torch.einsum("...bac,...dae->...bdce", target_feature.unsqueeze(-2), drug_feature.unsqueeze(-2))
        # [*, N_res, N_atom, hidden_dim * hidden_dim]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, N_res, N_atom, interaction_dim]
        outer = self.linear_out(outer)

        return outer

    def forward(self, target_embedding, drug_embedding, target_mask, drug_mask):
        """
        Args:
            target_embedding: [Batch_size, N_res, target_dim] target embedding
            drug_embedding: [Batch_size, N_atom, drug_dim] drug embedding
            target_mask: [Batch_size, N_res]  mask for each target residues
            drug_mask: [Batch_size, N_atom]  mask for each drug atoms
        Returns:
            [Batch_size, N_res, N_atom, interaction_dim] Interaction embedding for each drug-target pair
        """
        # [Batch_size, N_res, target_dim]
        target_embedding = self.layernorm_target(target_embedding)
        # [Batch_size, N_atom, drug_dim]
        drug_embedding = self.layernorm_drug(drug_embedding)

        # [Batch_size, N_res, hidden_dim]
        target_feature = self.linear_target(target_embedding) * target_mask.unsqueeze(-1)
        # [Batch_size, N_atom, hidden_dim]
        drug_feature = self.linear_drug(drug_embedding) * drug_mask.unsqueeze(-1)

        # [Batch_size, N_res, N_atom, interaction_dim]
        outer = self.opm(target_feature, drug_feature)

        # [Batch_size, N_res, N_atom]
        mask = torch.einsum("ab,ad->abd", target_mask, drug_mask).float()

        # [Batch_size, N_res, N_atom, interaction_dim] 这里不同于MSA里有个seq维度，这里维度是1因此求平均没有意义
        outer = outer * mask.unsqueeze(-1)

        return outer, mask


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11(True) and 12(Flase).
    """
    def __init__(self, interaction_dim, hidden_dim, outgoing=True):
        """
        Args:
            interaction_dim: Input channel dimension
            hidden_dim: Hidden channel dimension
            outgoing: from the outgoing side or not
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.interaction_dim = interaction_dim
        self.hidden_dim = hidden_dim
        self.outgoing = outgoing
        # projection linear and gate linear for edge between a and b
        self.linear_ap = nn.Linear(interaction_dim, hidden_dim)
        self.linear_ag = nn.Linear(interaction_dim, hidden_dim)

        self.linear_bp = nn.Linear(interaction_dim, hidden_dim)
        self.linear_bg = nn.Linear(interaction_dim, hidden_dim)

        self.linear_g = nn.Linear(interaction_dim, interaction_dim)
        self.linear_z = nn.Linear(hidden_dim, interaction_dim)

        self.layernorm_in = nn.LayerNorm(interaction_dim)
        self.layernorm_out = nn.LayerNorm(hidden_dim)

        self.sigmoid = nn.Sigmoid()

    def combine_projections(self, a, b):
        if self.outgoing:
            # [Batch_size, hidden_dim, N_res, N_res]
            p = torch.matmul(
                permute_final_dims(a, (2, 0, 1)), # [Batch_size, hidden_dim, N_res, N_atom]
                permute_final_dims(b, (2, 1, 0)), # [Batch_size, hidden_dim, N_atom, N_res]
            )
        else:
            # [Batch_size, hidden_dim, N_atom, N_atom]
            p = torch.matmul(
                permute_final_dims(a, (2, 1, 0)), # [Batch_size, hidden_dim, N_atom, N_res]
                permute_final_dims(b, (2, 0, 1)), # [Batch_size, hidden_dim, N_res, N_atom]
            )

        # [Batch_size, N_res, N_res, hidden_dim] for outgoing
        # [Batch_size, N_atom, N_atom, hidden_dim] for incoming
        return permute_final_dims(p, (1, 2, 0))

    def forward(self, interaction_embedding, mask):
        """
        Args:
            interaction_embedding: [Batch_size, N_res, N_atom, interaction_dim] interaction embedding input
            mask: [Batch_size, N_res, N_atom, 1] interaction mask input
        Returns:
            [Batch_size, N_res, N_atom, interaction_dim] update the interaction embedding
        """
        interaction_embedding = self.layernorm_in(interaction_embedding)

        # [Batch_size, N_res, N_atom, interaction_dim]
        a = self.linear_ap(interaction_embedding) * self.sigmoid(self.linear_ag(interaction_embedding))
        a = a * mask

        # [Batch_size, N_res, N_atom, interaction_dim]
        b = self.linear_bp(interaction_embedding) * self.sigmoid(self.linear_bg(interaction_embedding))
        b = b * mask

        # the core difference is in or out
        x = self.combine_projections(a, b)
        x = self.layernorm_out(x)
        x = self.linear_z(x)
        # [Batch_size, N_res, N_atom, interaction_dim]
        g = self.sigmoid(self.linear_g(interaction_embedding))
        if self.outgoing:
            z = torch.matmul(permute_final_dims(x, (2, 0, 1)), permute_final_dims(g, (2, 0, 1)))
        else:
            z = torch.matmul(permute_final_dims(g, (2, 0, 1)), permute_final_dims(x, (2, 0, 1)))
        return permute_final_dims(z, (1, 2, 0))


class TriangleAttention(nn.Module):
    def __init__(self, interaction_dim, hidden_dim, num_heads, starting=True, inf=1e9):
        """
        Args:
            interaction_dim: Input channel dimension
            hidden_dim: Overall hidden channel dimension (not per-head)
            num_heads: Number of attention heads
            starting: from the starting node or not
        """
        super(TriangleAttention, self).__init__()

        self.interaction_dim = interaction_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(interaction_dim)
        self.linear = nn.Linear(interaction_dim, num_heads, bias=False)
        self.mha = MultiHeadAttention(interaction_dim, interaction_dim, interaction_dim, hidden_dim, num_heads)


    def forward(self, interaction_embedding, mask):
        """
        Args:
            interaction_embedding: [Batch_size, N_res, N_atom, interaction_dim] interaction embedding input
            mask: [Batch_size, N_res, N_atom, 1] interaction mask input
        Returns:
            [Batch_size, N_res, N_atom, interaction_dim] update the interaction embedding
        """
        # Shape annotations assume self.starting. Else, I and J are flipped
        if not self.starting:
            interaction_embedding = interaction_embedding.transpose(-2, -3)
            mask = mask.transpose(-2, -3)

        # [Batch_size, N_res, N_atom, interaction_dim]
        interaction_embedding = self.layer_norm(interaction_embedding)
        # [Batch_size, N_res, 1, 1, N_atom]
        mask_bias = (self.inf * (mask.squeeze(-1) - 1))[..., :, None, None, :]
        # [Batch_size, 1, num_heads, N_res, N_atom]
        triangle_bias = permute_final_dims(self.linear(interaction_embedding), (0, 2, 1)).unsqueeze(-2)
        # 创新点的位置，可以再加入能量约束，例如自由能的预测值（interaction_embedding的线性输出）
        biases = [mask_bias, triangle_bias]

        interaction_embedding = self.mha(interaction_embedding, interaction_embedding, biases)

        if not self.starting:
            interaction_embedding = interaction_embedding.transpose(-2, -3)

        return interaction_embedding


class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """
    def __init__(self, interaction_dim, factor):
        """
        Args:
            hidden_dim: Pair interaction dimension
            factor: Factor by which hidden_dim is multiplied
        """
        super(PairTransition, self).__init__()
        self.interaction_dim = interaction_dim
        self.factor = factor

        self.layernorm = nn.LayerNorm(interaction_dim)
        self.linear_input = nn.Linear(interaction_dim, factor * interaction_dim)
        self.relu = nn.ReLU()
        self.linear_output = nn.Linear(factor * interaction_dim, interaction_dim)

    def transition(self, interaction_embedding, mask):
        # [Batch_size, N_res, N_atom, interaction_dim * factor]
        interaction_embedding = self.linear_input(interaction_embedding)
        interaction_embedding = self.relu(interaction_embedding)

        # [Batch_size, N_res, N_atom, interaction_dim]
        interaction_embedding = self.linear_output(interaction_embedding) * mask

        return interaction_embedding

    def forward(self, interaction_embedding, mask=None):
        """
        Args:
            interaction_embedding: [Batch_size, N_res, N_atom, interaction_dim] interaction embedding input
            mask: [Batch_size, N_res, N_atom, 1] interaction mask input
        Returns:
            [Batch_size, N_res, N_atom, interaction_dim] update the interaction embedding
        """
        # [Batch_size, N_res, N_atom, interaction_dim]
        interaction_embedding = self.layernorm(interaction_embedding)

        interaction_embedding = self.transition(interaction_embedding, mask)

        return interaction_embedding

class PairStack(nn.Module):

    def __init__(self, d_pair, p_drop=0.25):
        super(PairStack, self).__init__()

        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(d_pair)
        self.TriangleMultiplicationIncoming = TriangleMultiplicationIncoming(d_pair, p_drop=p_drop)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(d_pair, p_drop=p_drop)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(d_pair, p_drop=p_drop)
        self.PairTransition = Transition(d=d_pair)

    def forward(self, pair):
        pair = self.TriangleMultiplicationOutgoing(pair)
        pair = self.TriangleMultiplicationIncoming(pair)
        pair = self.TriangleAttentionStartingNode(pair)
        pair = self.TriangleAttentionEndingNode(pair)
        pair = self.PairTransition(pair)
        return pair

