import math
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils, data, models
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torch_cluster import nearest
from torch_scatter import scatter_mean, scatter_add
from pykeops.torch import LazyTensor

@R.register("tasks.PropertyPrediction")
class PropertyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule / protein property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=1,
                 normalization=True, num_class=None, graph_construction_model=None, verbose=0):
        super(PropertyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)])

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            # My code for MSE metric, a little diffenent from the rmse without the sqrt()
            elif _metric == "mse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            # My code for rm2 metric from DeepDTA code
            elif _metric == "rm2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.rm2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)                
            # My code for C-Index metric
            elif _metric == "c_index":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.c_index(_pred[_labeled], _target[_labeled])
                    score.append(_score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric


@R.register("tasks.MultipleBinaryClassification")
class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        # self.trans_layer = models.ESM(path="../../result/model_pth/ESM/", model="ESM-2_650M")  # added for residue_feature from 21 to 1280
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        # esm_feature = self.trans_layer(graph, graph.residue_feature.float())['residue_feature'] 
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)  # graph.node_feature.float()
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.NodePropertyPrediction")
class NodePropertyPrediction(tasks.Task, core.Configurable):
    """
    Node / atom / residue property prediction task.

    Parameters:
        model (nn.Module): graph representation model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
            Available entities are ``node``, ``atom`` and ``residue``.
        num_class (int, optional): number of classes
        verbose (int, optional): output verbose level
    """

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("macro_auprc", "macro_auroc"), num_mlp_layer=1,
                 normalization=True, num_class=None, verbose=0):
        super(NodePropertyPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation on the training set.
        """
        self.view = getattr(train_set[0]["graph1"], "view", "atom")
        values = torch.cat([data["graph1"].target for data in train_set])
        mean = values.float().mean()
        std = values.float().std()
        if values.dtype == torch.long:
            num_class = values.max().item()
            if num_class > 1 or "bce" not in self.criterion:
                num_class += 1
        else:
            num_class = 1

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [self.num_class])

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph1"]
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.view in ["node", "atom"]:
            output_feature = output["node_feature"]
        else:
            output_feature = output.get("residue_feature", output.get("node_feature"))
        pred = self.mlp(output_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred.squeeze(dim=-1)

    def target(self, batch):
        size = batch["graph1"].num_nodes if self.view in ["node", "atom"] else batch["graph1"].num_residues
        return {
            "label": batch["graph1"].target,
            "mask": batch["graph1"].mask,
            "size": size
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        labeled = ~torch.isnan(target["label"]) & target["mask"]

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"].float(), reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target["label"], reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target):
        metric = {}
        _target = target["label"]
        _labeled = ~torch.isnan(_target) & target["mask"]
        _size = functional.variadic_sum(_labeled.long(), target["size"]) 
        for _metric in self.metric:
            if _metric == "micro_acc":
                score = metrics.accuracy(pred[_labeled], _target[_labeled].long())
            elif metric == "micro_auroc":
                score = metrics.area_under_roc(pred[_labeled], _target[_labeled])
            elif metric == "micro_auprc":
                score = metrics.area_under_prc(pred[_labeled], _target[_labeled])
            elif _metric == "macro_auroc":
                score = metrics.variadic_area_under_roc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_auprc":
                score = metrics.variadic_area_under_prc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_acc":
                score = pred[_labeled].argmax(-1) == _target[_labeled]
                score = functional.variadic_mean(score.float(), _size).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric

@R.register("tasks.InteractionPrediction")
class InteractionPrediction(PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        model (nn.Module, or list of nn.Moudle): graph representation model(s) for drug
        model2 (nn.Module, or list of nn.Moudle): graph representation model(s) for protein
        complex_model (nn.Module, or list of nn.Moudle): graph representation model(s) for complex graph
        model_type(str): 'drug' for single drug model, 'target' for single target model, 'dual-drug' for two drug model
            and one target model, and so on.
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``mse``, ``c_index``, ``r2``,``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        hidden_dim (int, optional): the dim for drug and protein attention linear transformations
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        graph_construction_model (nn.Module, optional): graph construction model for protein/complex
        graph_construction_model2 (nn.Module, optional): graph construction model for drug
        **kwargs
    """
    def __init__(
            self, model, model2, mode_type=None, task=(), criterion="mse", metric=("mse", "c_index", "r2"),
            num_mlp_layer=3, hidden_dim=256, normalization=True, num_class=None,
            graph_construction_model=None, graph_construction_model2=None, **kwargs):
        super(InteractionPrediction, self).__init__(model, **kwargs)
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.mode = mode_type  # mode needed to be update
        self.target_model = model
        self.drug_model = model2 or model
        self.graph_construction_model = graph_construction_model  # added
        self.graph_construction_model2 = graph_construction_model2  # added
        self.hidden_dim = hidden_dim  # protein and drug representation final dims
        self.readout_drug = layers.MeanReadout(type='node')
        self.readout_protein = layers.MeanReadout(type='residue')
        # self.mdn_block = MDB_Block(
        #     target_output_dim=self.target_model.output_dim,
        #     drug_output_dim=self.drug_model.output_dim,
        #     hidden_dim=128
        # )
        #

        self.target_seq_linear = layers.MLP(
            1280, [hidden_dim], batch_norm=True, dropout=0.2
        )
        self.target_str_linear = layers.MLP(
            self.target_model.output_dim, [hidden_dim], batch_norm=True, dropout=0.2
        )

        self.drug_seq_linear = layers.MLP(
            768, [hidden_dim], batch_norm=True, dropout=0.2
        )
        self.drug_str_linear = layers.MLP(
            self.drug_model.output_dim, [hidden_dim], batch_norm=True, dropout=0.2
        )
        self.deep_fusion = DeepFusion(input_dim=hidden_dim, hidden_dim=64)


    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                # if not math.isnan(sample[task]):  # error! only scalar, but tensor here for all residue label
                values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            # addad for skip the pocket label tensor mean/std computation
            if task == 'pocket':
                continue
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class
        if self.mode == 'DTA':  # added TransformerCPI method
            self.predictor = models.DTAPredictor(target_dim=self.target_model.output_dim,
                                                 drug_dim=self.drug_model.output_dim, hidden_dim=self.hidden_dim)
        elif self.mode == 'MolFormer':
            hidden_dims = [self.target_model.output_dim] * (self.num_mlp_layer - 1)
            # hidden_dims = [1024, 512, 256]
            self.mlp = layers.MLP(self.target_model.output_dim + self.drug_model.output_dim + 768 +1280,  # self.target_model.output_dim + self.drug_model.output_dim +
                                  hidden_dims + [sum(self.num_class)], batch_norm=True)
        elif self.mode == 'InteractionMap':
            hidden_dims = [self.target_model.output_dim] * (self.num_mlp_layer - 1)
            interaction_dim = 128
            self.interaction_init = InteractionMapInit(self.target_model.output_dim, self.drug_model.output_dim, hidden_dim=interaction_dim)
            self.outgoing_update = TriangleMultiplicativeUpdate(input_dim=interaction_dim, hidden_dim=interaction_dim, outgoing=True)
            self.incoming_update = TriangleMultiplicativeUpdate(input_dim=interaction_dim, hidden_dim=interaction_dim, outgoing=False)
            self.transition_update = layers.MLP(input_dim=interaction_dim, hidden_dims=[interaction_dim * 2, interaction_dim], batch_norm=True)
            self.mlp = layers.MLP(self.target_model.output_dim + self.drug_model.output_dim + interaction_dim * 2,
                                  hidden_dims + [sum(self.num_class)], batch_norm=True)
        elif self.mode == 'DeepFusion':
            hidden_dims = [self.hidden_dim] * (self.num_mlp_layer - 1)
            self.mlp = layers.MLP(self.hidden_dim * 2, hidden_dims + [sum(self.num_class)], batch_norm=True)
        elif self.mode == 'ComplexGraph':
            hidden_dims = [self.target_model.output_dim] * (self.num_mlp_layer - 1)
            self.mlp = layers.MLP(self.complex_model.output_dim, hidden_dims + [sum(self.num_class)], batch_norm=True)
        elif self.mode == "KIBA":
            self.mlp = layers.MLP(self.hidden_dim * 2, [1024, 256, 1], dropout=0.2)
        else:  # original method!
            hidden_dims = [self.target_model.output_dim] * (self.num_mlp_layer - 1)
            self.mlp = layers.MLP(self.target_model.output_dim + self.drug_model.output_dim,
                                  hidden_dims + [sum(self.num_class)], batch_norm=True)


    def predict(self, batch, all_loss=None, metric=None):
        if self.mode == 'DeepFusion':
            # 分别融合序列和结构特征
            # ===== Protein Representation Learning ===== #
            graph1 = batch["graph1"]
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            # protein_seq = self.readout_protein(graph1, graph1.residue_feature)
            protein_seq = graph1.residue_feature
            protein_str = output1["node_feature"]
            # protein_str = output1["graph_feature"]
            # ===== Protein Residue-Level Fusion ===== #
            protein_seq = self.target_seq_linear(protein_seq)
            protein_str =self.target_str_linear(protein_str)
            protein_feature = self.deep_fusion(torch.stack([protein_seq, protein_str], dim=1))
            protein_feature = self.readout_protein(graph1, protein_feature)
            # ===== Drug feature ===== #
            graph2 = batch["graph2"]
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            drug_seq = graph2.mol_feature
            drug_str = output2["graph_feature"]
            # ===== Drug Graph-Level Fusion ===== #
            drug_seq = self.drug_seq_linear(drug_seq)
            drug_str = self.drug_str_linear(drug_str)
            drug_feature = self.deep_fusion(torch.stack([drug_seq, drug_str], dim=1))
            # ===== Final MLP Prediciton ===== #
            pred = self.mlp(torch.cat([protein_feature, drug_feature], dim=-1))
        elif self.mode == 'PSSA':
            # 以残基和原子的形似融合特征
            # ===== Protein Residue feature ===== # Input Shape (14536, 14536, 7) in batch variadic form
            graph1 = batch["graph1"]
            if self.graph_construction_model:  # added
                graph1 = self.graph_construction_model(graph1)  # added
            output1  = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            protein_output = layers.functional.variadic_to_padded(output1["node_feature"], graph1.num_residues)[0]  # Shape (128, 326, 512)
            protein_output = self.protein_linear(protein_output)  # linear transformation into DT space Shape (128, 326, 256)
            # ===== Drug Atom feature ===== # Shape (4333, 4333, 4)  46 atoms for example
            graph2 = batch["graph2"]
            if self.graph_construction_model2:  # added
                graph2 = self.graph_construction_model2(graph2)  # added
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            drug_output = layers.functional.variadic_to_padded(output2["node_feature"], graph2.num_nodes)[0] # Shape (128, 46, 256)
            drug_output = self.drug_linear(drug_output)  # linear transformation into DT space Shape (128, 46, 256)
            drug_feature = layers.functional.padded_to_variadic(drug_output, graph2.num_nodes)
            drug_feature = self.readout_drug(graph2, drug_feature)
            drug_feature = torch.cat([graph2.mol_feature, drug_feature], dim=-1)
            # ===== Protein Fusion ===== # T: (128, 1024)
            protein_feature = self.pssa_protein(protein_output, drug_output, drug_output)  # cross attention
            protein_feature = layers.functional.padded_to_variadic(protein_feature, graph1.num_residues)
            protein_feature = torch.cat([self.readout_protein(graph1, graph1.residue_feature), self.readout_protein(graph1, protein_feature)], dim=-1)
            # ===== Final MLP Prediciton ===== #
            pred = self.mlp(torch.cat([protein_feature, drug_feature], dim=-1))
        elif self.mode == 'DTA':
            graph1 = batch["graph1"]
            if self.graph_construction_model:  # added
                graph1 = self.graph_construction_model(graph1)  # added
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            graph2 = batch["graph2"]
            if self.graph_construction_model2:  # added
                graph2 = self.graph_construction_model(graph2)  # added
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            # TODO: DTAPredictor Demo
            pred = self.predictor(output1["node_feature"], output2["node_feature"], graph1, graph2)
        elif self.mode == 'MolFormer':
            graph1 = batch["graph1"]
            # residue_type = layers.functional.one_hot(graph1.residue_type, len(data.Protein.residue2id) + 1)
            # output1 = self.target_model(graph1, residue_type.float(), all_loss=all_loss, metric=metric)
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            protein_output = torch.cat(
                [self.readout_protein(graph1, graph1.residue_feature), output1["graph_feature"]], dim=-1
            )
            # protein_output = self.readout_protein(graph1, graph1.residue_feature)
            # protein_output = output1["graph_feature"]
            graph2 = batch["graph2"]
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            drug_output = torch.cat([graph2.mol_feature, output2["graph_feature"]], dim=-1)
            # drug_output = graph2.mol_feature
            # drug_output = output2["graph_feature"]
            pred = self.mlp(torch.cat([protein_output, drug_output], dim=-1))
        elif self.mode == "InteractionMap":
            # ========== Internal Interaction Stage ========== #
            target_graph = batch["graph1"]
            target_feature = self.target_model(target_graph, target_graph.node_feature.float(), all_loss=all_loss, metric=metric) # [Num_residue, target_output]
            drug_graph = batch["graph2"]
            drug_feature = self.drug_model(drug_graph, drug_graph.node_feature.float(), all_loss=all_loss, metric=metric) # [Num_atom, drug_output]
            # ========== External Interaction Stage ========== #
            interaction_map = self.interaction_init(target_graph, drug_graph, target_feature, drug_feature)  # [Num_residue, Num_atom, hidden_dim]
            interaction_map = self.outgoing_update(interaction_map)
            interaction_map = self.incoming_update(interaction_map)
            interaction_map = self.transition_update(interaction_map)  # [Num_residue, Num_atom, hidden_dim]
            # ========== Fusion Prediction Stage ========== #
            target_inter = target_feature["graph_feature"]
            target_exter = self.readout_protein(target_graph, torch.mean(interaction_map, dim=1))
            target_output = torch.cat([target_inter, target_exter], dim=-1)
            drug_inter = drug_feature["graph_feature"]
            drug_exter = self.readout_drug(drug_graph, torch.mean(interaction_map, dim=0))
            drug_output = torch.cat([drug_inter, drug_exter], dim=-1)
            pred = self.mlp(torch.cat([target_output, drug_output], dim=-1))
        elif self.mode == 'ComplexGraph':
            # ========== Internal Graph GNN Stage ========== #
            graph1 = batch["graph1"]
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            graph2 = batch["graph2"]
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            # ========== Complex Interaction Graph Construction ========== #
            complex_list = []
            if self.graph_construction_model:
                for index in range(len(graph1)):
                    complex_graph = data.Molecule.pack([graph2[index], graph1[index]])
                    complex_graph = complex_graph.merge([0, 0])  # merge two graph into one single graph
                    complex_list.append(self.graph_construction_model(complex_graph))
            complex_packed = data.Molecule.pack(complex_list)
            # ========== Complex Graph GNN Stage ========== #
            output = self.complex_model(complex_packed, complex_packed.node_feature.float(), all_loss=all_loss, metric=metric)
            pred = self.mlp(output['graph_feature'])
        elif self.mode == 'KIBA':
            graph1 = batch["graph1"]
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            protein_str = self.target_str_linear(output1["graph_feature"])
            graph2 = batch["graph2"]
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            drug_str = self.drug_str_linear(output2["graph_feature"])
            pred = self.mlp(torch.cat([protein_str, drug_str], dim=-1))
        else: # original method!
            graph1 = batch["graph1"]
            if self.graph_construction_model:  # added
                graph1 = self.graph_construction_model(graph1)  # added
            output1 = self.target_model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
            graph2 = batch["graph2"]
            if self.graph_construction_model2:  # added
                graph2 = self.graph_construction_model2(graph2)  # added
            output2 = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
            pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


@R.register("tasks.MoleculeDocking")
class MoleculeDocking(tasks.Task, core.Configurable):
    """
    Predict the Molecule Docking coords in the complex.

    Parameters:
        model (nn.Module, or list of nn.Moudle): EGNN model(s) for drug coords update
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s), available criterions are ``rmsd``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``rmsd``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        hidden_dim (int, optional): the dim for drug and protein attention linear transformations

        **kwargs
    """
    def __init__(self, model, task=(), criterion=["RMSD", "MSE"], metric=("RMSD"), length_weight=1,**kwargs):
        super(MoleculeDocking, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.length_weight = length_weight

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        updated_Mols = self.predict(batch)
        original_Mols = self.target(batch)
        # ============================== Min RMSD ============================== #
        all_loss += self.kabschRMSD(updated_Mols, original_Mols)
        metric[self.criterion[0]] = all_loss
        # ============================== Bond Constraint ============================== #
        bond_length_loss = self.bondConstraint(updated_Mols, original_Mols) * self.length_weight
        metric[self.criterion[1]] = bond_length_loss
        all_loss += bond_length_loss
        return all_loss, metric
            
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph2"]
        self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)  # update the coords
        return graph  # updated_coord

    def target(self, batch):
        graph = batch["graph3"]
        return graph
    
    def evaluate(self, pred, target):
        metric = {}
        # ============================== Min RMSD ============================== #
        metric[self.metric] = self.kabschRMSD(pred, target)

        return metric

    def normalRMSD(self, updated_Mols, original_Mols):
        coord_diff = (original_Mols.node_position - updated_Mols.node_position) ** 2
        msd = scatter_mean(coord_diff.sum(dim=-1), original_Mols.atom2graph, dim=0, dim_size=len(original_Mols))
        return torch.sqrt(msd).mean()

    def kabschRMSD(self, updated_Mols, original_Mols):
        rmsds = []
        for updated_Mol, original_Mol in zip(updated_Mols, original_Mols):
            lig_coords_pred = updated_Mol.node_position
            lig_coords = original_Mol.node_position
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.stack(rmsds).mean()

    def bondConstraint(self, updated_Mols, original_Mols):
        node_in, node_out, _ = original_Mols.edge_list.t()
        original_length = torch.norm(
            (original_Mols.node_position[node_out] - original_Mols.node_position[node_in]), dim=-1)
        updated_length = torch.norm(
            (updated_Mols.node_position[node_out] - updated_Mols.node_position[node_in]), dim=-1)
        return F.mse_loss(updated_length, original_length, reduction="none").mean()


@R.register("tasks.AffinityWithDocking")
class AffinityWithDocking(tasks.Task, core.Configurable):
    """
    Predict the Drug-Target Affinity and Molecule Docking coords in the same time by EGNN.

    Parameters:
        model (nn.Module, or list of nn.Moudle): EGNN model(s) for drug coords update
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s), available criterions are ``rmsd``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``rmsd``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        hidden_dim (int, optional): the dim for drug and protein attention linear transformations

        **kwargs
    """

    def __init__(self, target_model, drug_model, task, point_model=None, complex_model=None, criterion="mse",
                 criterion_weight={"Affinity MSE": 1}, mode_type=None,
                 metric=("mse", "c_index", "r2"), num_mlp_layer=3, hidden_dim=256, normalization=True,
                 graph_construction_model=None, **kwargs):
        super(AffinityWithDocking, self).__init__()
        self.drug_model = drug_model
        self.target_model = target_model
        self.point_model = point_model
        self.complex_model = complex_model
        self.graph_construction_model = graph_construction_model  # added

        self.task = task
        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.mode_type = mode_type
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.hidden_dim = hidden_dim
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)

        self.linear_protein = layers.MLP(self.target_model.output_dim, self.drug_model.output_dim)
        self.pocket_classifier = nn.Sequential(
            nn.Linear(self.point_model.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.readout_protein = layers.MeanReadout(type='node')
        self.readout_point = layers.MaxReadout(type='node')
        # self.mdn_block = MDB_Block(self.target_model.output_dim, self.drug_model.output_dim)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            for task in self.task:
                values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task in self.task:
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = 1

        hidden_dims = [self.hidden_dim] * (self.num_mlp_layer - 1)
        if self.complex_model is not None:
            self.mlp = layers.MLP(self.complex_model.output_dim, hidden_dims + [self.num_class], batch_norm=True)
        elif self.mode_type == "MolFormer":
            self.readout_protein = layers.MeanReadout(type='residue')
            hidden_dims = [self.target_model.output_dim] * (self.num_mlp_layer - 1)
            self.mlp = layers.MLP(
                self.target_model.output_dim + self.drug_model.output_dim + self.point_model.output_dim + 768 + 1280,
                hidden_dims + [self.num_class], batch_norm=True)
        else:
            self.mlp = layers.MLP(self.target_model.output_dim + self.drug_model.output_dim,
                                  hidden_dims + [self.num_class], batch_norm=True)

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0
        # ========== RMSD Loss ========== #
        if "RMSD" in self.criterion_weight:
            rmsd = self.kabschRMSD(self.updated_Mols, self.original_Mols)
            metric["RMSD"] = rmsd
            all_loss += rmsd * self.criterion_weight["RMSD"]
        # ========== Bond MSE Loss ========== #
        if "Bond MSE" in self.criterion_weight:
            bond_length_mse = self.bondConstraint(self.updated_Mols, self.original_Mols)
            metric["Bond MSE"] = bond_length_mse
            all_loss += bond_length_mse * self.criterion_weight["Bond MSE"]
        # ========== MDN Loss ========== #
        # mdn_loss = self.mdn_block.mdn_loss(self.pi, self.sigma, self.mu, self.distance)
        # metric["MDN Loss"] = mdn_loss
        # all_loss += mdn_loss * self.criterion_weight["MDN Loss"]
        # ========== Affinity MSE Loss ========== #
        affinity_mse = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
        affinity_mse = functional.masked_mean(affinity_mse, labeled, dim=0).sum()
        metric["Affinity MSE"] = affinity_mse
        all_loss += affinity_mse * self.criterion_weight["Affinity MSE"]

        # metric["TOTAL LOSS"] = all_loss
        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        graph2 = batch["graph2"]
        # ========== Protein Pocket Stage ========== #
        residue_type = graph1.residue_type  # [N_residue, ]
        residue_type = layers.functional.one_hot(residue_type, len(data.Protein.residue2id))  # [N_residue, 20] no others
        pocket_index, target_point_features = self.point_model(graph1, residue_type.float())
        # torch.cuda.empty_cache()
        # ========== Internal Graph GNN Stage ========== #
        pocket = graph1.residue_mask(index=pocket_index)
        target_structure_features = self.target_model(pocket, pocket.node_feature.float(), all_loss=all_loss, metric=metric)
        drug_structure_features = self.drug_model(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        # self.updated_Mols = graph2

        if self.mode_type == "MolFormer":
            target_sequence_features = self.readout_protein(pocket, pocket.residue_feature)
            target_point_features = self.readout_protein(pocket, target_point_features)
            protein_output = torch.cat([
                target_sequence_features,  # sequence feature
                target_structure_features["graph_feature"],  # structure feature
                target_point_features],  # surface feature
                dim=-1
            )
            if torch.isnan(protein_output).any():
                print("Nan in Pred")
                protein_output = torch.where(
                    torch.isnan(protein_output), torch.full_like(protein_output, 0), protein_output)  # add for nan
            drug_sequence_features = graph2.mol_feature
            drug_output = torch.cat([
                drug_sequence_features,  # sequence feature
                drug_structure_features["graph_feature"]  # structure feature
            ], dim=-1)
            # ========== Distance Prediction with MDN from Node Feature ========== #
            # self.pi, self.sigma, self.mu, self.distance = self.mdn_block(
            #     pocket_graph=pocket, drug_graph=graph2,
            #     pocket_feature=target_structure_features["node_feature"],
            #     drug_feature=drug_structure_features["node_feature"]
            # )
            pred = self.mlp(torch.cat([protein_output, drug_output], dim=-1))
        else:
            pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean

        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)  # 改成shape一致的[128, 1]
        if "graph3" in batch:
            graph3 = batch["graph3"]
            self.original_Mols = graph3
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)
        metric = {}

        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "mse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "c_index":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.c_index(_pred[_labeled], _target[_labeled])
                    score.append(_score)
            elif _metric == "rm2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.rm2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s
        return metric

    def normalRMSD(self, updated_Mols, original_Mols):
        coord_diff = (original_Mols.node_position - updated_Mols.node_position) ** 2
        msd = scatter_mean(coord_diff.sum(dim=-1), original_Mols.atom2graph, dim=0, dim_size=len(original_Mols))
        return torch.sqrt(msd).mean()

    def kabschRMSD(self, updated_Mols, original_Mols):
        rmsds = []
        for updated_Mol, original_Mol in zip(updated_Mols, original_Mols):
            lig_coords_pred = updated_Mol.node_position
            lig_coords = original_Mol.node_position
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.stack(rmsds).mean()

    def bondConstraint(self, updated_Mols, original_Mols):
        node_in, node_out, _ = original_Mols.edge_list.t()
        original_length = torch.norm(
            (original_Mols.node_position[node_out] - original_Mols.node_position[node_in]), dim=-1)
        updated_length = torch.norm(
            (updated_Mols.node_position[node_out] - updated_Mols.node_position[node_in]), dim=-1)
        return F.mse_loss(updated_length, original_length, reduction="none").mean()

    def smooth_distance_function(self, molecule, point, smoothness=0.2, k=128):
        """
        :param
            molecule:  molecule the atom are Red circulars in Figure.

            point: surface cloud points of protein are Blue dots in Figure.
        Returns:
            Tensor: computed smooth distances between sampled points and target surface.

        Computes a smooth distance from protein points to the center of atoms of a drug.
        Implements Formula:
        SDF(x) = -(B/C) * D, where
        - B = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)*σₖ
        - C = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)
        - D = logΣₖ₌₁ᴬ exp(-‖x-aₖ‖/σₖ)
        """
        # residue_radii = torch.Tensor(
        #     [1.46, 1.54, 1.64, 1.88, 1.99, 1.61, 1.80, 1.99, 2.02, 1.83,
        #      1.88, 1.96, 2.17, 2.07, 2.05, 2.02, 2.18, 2.17, 2.19, 2.38,
        #      1.94]).unsqueeze(dim=1).cuda()

        # atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
        atom_radii = torch.Tensor(
            [1.10, 1.92, 1.70, 1.55, 1.52, 1.47, 1.73, 2.10, 1.80, 1.80,
             1.75, 1.40, 1.39, 1.90, 1.85, 2.17, 1.98]
        ).unsqueeze(dim=1).cuda()

        # ========== Index Stage ========== #
        # 得到点云和原子的坐标，方便计算点云到表面距离来作为约束
        molecule_coords = molecule.node_position
        point_coords = point.node_position
        # 药物分子的中心坐标
        center_coords = molecule_coords.mean(dim=0, keepdim=True)
        # 计算目标点与所有点云之间的距离
        distances = torch.norm(point_coords - center_coords, dim=1)
        # 获取按距离递增排序的索引
        sorted_indices = torch.argsort(distances)
        # 获取最近的k个点的索引
        closest_indices = sorted_indices[:k]
        # 根据索引获取最近的100个点的坐标
        closest_points = point_coords[closest_indices]
        # 计算采样的这k个点到每个原子的向量与点云表面法向量之间的关系（相同说明原子在蛋白质表面外，否则就在内部即发生碰撞）
        diff_coords = molecule_coords.unsqueeze(dim=1) - closest_points.unsqueeze(dim=0)  # [atom_nums, k ,3]
        point_normals = point.node_normal[closest_indices]
        side = torch.sum(diff_coords * point_normals, dim=-1) > 0
        inside_counts = (side == False).sum() / (side.shape[0] * side.shape[1])

        # ========== Distance Stage ========== #
        # ‖x-aₖ‖ is the distance between all residue a_k and atom x
        distance_matrix = (
                (molecule_coords.unsqueeze(dim=1) - closest_points.unsqueeze(dim=0)) ** 2
        ).sum(dim=-1).sqrt()
        # σₖ is the radii of the current residue with smoothness
        atom_types = molecule.node_feature[:, :17].to(torch.float)
        atomtype_radii = atom_types @ atom_radii  # (N, 17) @ (17, 1) -> (N, 1)
        atomtype_radii = smoothness * atomtype_radii
        # B = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)*σₖ
        B = ((-distance_matrix).exp() * atomtype_radii).sum(dim=0)
        # C = Σₖ₌₁ᴬ exp(-‖x-aₖ‖)
        C = (-distance_matrix).exp().sum(dim=0)
        # D = logΣₖ₌₁ᴬ exp(-‖x-aₖ‖/σₖ)
        D = (-distance_matrix / atomtype_radii).logsumexp(dim=0)

        smooth_distance = -(B / C) * D

        return smooth_distance, closest_indices, inside_counts


# 通过EGNN得到的距离，与特征对相互约束，即通过特征也能推断出距离
class MDB_Block(nn.Module):
    def __init__(
            self, target_output_dim, drug_output_dim, hidden_dim=32, num_gaussians=10, dropout=0.10, threhold=7.0
    ):
        super(MDB_Block, self).__init__()
        self.threhold = threhold

        self.linear_target = nn.Linear(target_output_dim, hidden_dim)
        self.linear_drug = nn.Linear(drug_output_dim, hidden_dim)
        self.pair_linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout)
        )
        self.norm_out = nn.LayerNorm(hidden_dim + 1)
        self.linear_pi = nn.Linear(hidden_dim + 1, num_gaussians)
        self.linear_sigma = nn.Linear(hidden_dim + 1, num_gaussians)
        self.linear_mu = nn.Linear(hidden_dim + 1, num_gaussians)
        # Aux loss for Drug feature constraint
        # self.atom_classifier = nn.Linear(drug_output_dim, 17)
        # self.bond_classifier = nn.Linear(drug_output_dim * 2, 4)

    def forward(self, pocket_graph, drug_graph, pocket_feature, drug_feature):
        # [Done]这里的靶标通过具体的残基与药物原子之间的最小距离约束，然后再使用MDN进行相互学习并制约
        # ========== Interaction Map Init with Residue Feature and Atom Feature ========== #
        pocket_feature = self.linear_target(pocket_feature)  # [Num_residue, hidden_dim]
        drug_feature = self.linear_drug(drug_feature)  # [Num_atom, hidden_dim]
        # ========== Score Residue Feature and Atom Feature for Compact ========== #
        pocket_score = torch.sigmoid(pocket_feature)
        pocket_feature = pocket_feature * pocket_score
        drug_score = torch.sigmoid(drug_feature)
        drug_feature = drug_feature * drug_score
        # [TODO] 注意药物的坐标应该会更新(EGNN)！仅固定蛋白质坐标，会用keops节省显存
        # ========== Interaction Map Concat Features in Each Pair Range ========== #
        pocket_feature = pocket_feature.unsqueeze(dim=1)  # [Num_residue, 1, hidden_dim]
        drug_feature = drug_feature.unsqueeze(dim=0)  # [1, Num_atom, hidden_dim]
        pair_feature = pocket_feature + drug_feature # [Num_residue, Num_atom, hidden_dim]
        pocket_coord = pocket_graph.node_position
        drug_coord = drug_graph.node_position
        pair_distance = torch.cdist(pocket_coord, drug_coord).unsqueeze(dim=-1)
        pair_output = torch.cat([pair_feature, pair_distance], dim=-1)
        pair_output = self.norm_out(pair_output)
        # ========== MDN Params Computation ========== #
        pi = F.softmax(self.linear_pi(pair_output), -1)
        sigma = F.elu(self.linear_sigma(pair_output)) + 1.1
        mu = F.elu(self.linear_mu(pair_output)) + 1
        # ========== Drug Aux Task ========== #
        atom_types = self.atom_classifier(drug_output)
        node_in, node_out, relation = drug_graph.edge_list.t()
        bond_types = self.bond_classifier(torch.cat((drug_feature[node_in], drug_feature[node_out]), dim=1))
        return pi, sigma, mu, pair_distance, atom_types, bond_types

    def mdn_loss(self, pi, sigma, mu, distance):
        normal = torch.distributions.Normal(mu, sigma)
        loglike = normal.log_prob(distance.expand_as(normal.loc))
        mdn_loss = - torch.logsumexp(torch.log(pi) + loglike, dim=-1)
        index = torch.where(distance <= self.threhold)
        mdn_loss = mdn_loss[index[0], index[1]].mean()
        return mdn_loss

    def aux_loss(self, atom_types, bond_types, drug):
        atom_loss = F.cross_entropy(atom_types, drug.atom_type)
        bond_loss = F.cross_entropy(bond_types, drug.bond_type)
        return atom_loss + bond_loss

    def probablity_prediction(self, pi, sigma, mu, y):
        normal = torch.distributions.Normal(mu, sigma)
        logprob = normal.log_prob(y.expand_as(normal.loc))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)
        return prob


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(AttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_size = hidden_dim // num_heads

        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)
  
    def forward(self, query, key, value, mask=None):
        """
        query, key, value shape: (batch_size, hidden_dim) for drug and protein graph feature
        """
        num_batches = query.size(0)
        # split the hidden_dim into num_heads spaces
        key = self.linear_k(key).view(
            num_batches,
            -1,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        query = self.linear_q(query).view(
            num_batches,
            -1,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        value = self.linear_v(value).view(
            num_batches,
            -1,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)

        atten_score = self.attention(query, key, value, mask)
        # return the beginning shape (batch_size, hidden_dim)
        atten_score = atten_score.transpose(1, 2).contiguous().view(
            num_batches,
            self.hidden_dim
        )
        # returen to the space of drug and protein
        atten_score = self.linear_merge(atten_score)
        atten_score = atten_score.squeeze()
        return atten_score
    
    def attention(self, query, key, value, mask):
        scale = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(scale)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        atten_map = F.softmax(scores, dim=-1)  # 标签全是1
        # atten_map = F.sigmoid(scores) 
        atten_map = self.dropout(atten_map)

        return torch.matmul(atten_map, value)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ration=4):
        """
        channels is different feature form drug and protein, like 4
        """
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Linear(in_channels, in_channels // reduction_ration)
        self.excitation = nn.Linear(in_channels // reduction_ration, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        input: 
            shape (batch_size, channels, hidden_dim) 
            squeeze the hidden_dim into a global scale
        """
        batch_size, channels, _ = input.size()
        output = self.avgpool(input).view(batch_size, channels)
        output = self.squeeze(output)
        output = self.relu(output)
        output = self.excitation(output)
        output = self.relu(output)
        output = self.sigmoid(output).view(batch_size, channels, 1)
        return input * output


class concat_attention(nn.Module):
    """
    use Conv1d to get the attention of each position
    """
    def __init__(self):
        super(concat_attention, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)  # kernel_size(7, 15, 31)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        input = input.unsqueeze(dim=1)  # shape (4, 1, 64)
        enhanced_input = torch.cat((input, self.dropout(input)), dim=1)  # shape (4, 2, 64)  two input channel
        attention_weight = self.sigmoid(self.conv(enhanced_input))
        return attention_weight.squeeze(dim=1)  # shape (4, 64)


class ProbabilisticSparseAttention(nn.Module):
    """
    Probabilistic Sparse Self-Attention (PSSA) with positional bias.
        q, (k=v) represent a drug and protein. In Drug-Target Interaction, 
        Q maybe drug (batch_size, num_atom, hidden_dim).
        K maybe protein (batch_size, num_residue, hidden_dim).
        What we need is through different drug get current protein represention.
    """
    def __init__(self, embed_dim, num_heads=8, top_k=20, dropout=0.1):
        super(ProbabilisticSparseAttention, self).__init__()
        self.input_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = embed_dim // num_heads
        self.top_k = top_k  # top k value

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False) # qkv linear no bias!!!
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim)  # down-sampling
        # self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        batch_size, _, input_dim = k.size()
        # Linear transformations for query, key, and value and reshape (batch_size, num_heads, feature_length, hidden_dim)
        query = self.linear_q(q).view(batch_size, -1, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        key = self.linear_k(k).view(batch_size, -1, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        value = self.linear_v(v).view(batch_size, -1, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # further positional bias
        scores = self.softmax(scores)  # shape (32, 8, 1000, 46) protein residue representation to drug atom representation
        # Select top-k values for each query
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)  # dim=-1 shape (32, 8, 1000, 20) dim=-1 shape (32, 8, 20, 46)
        attn_weights = torch.zeros_like(scores)
        attn_weights.scatter_(-1, topk_indices, topk_scores)
        # Apply dropout
        # attn_weights = self.dropout(attn_weights)
        # Reshape and concatenate attention outputs
        attn_output = torch.matmul(attn_weights, value).permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, input_dim)
        # Linear transformation and residual connection
        output = self.output(attn_output) + q
        # output = self.output(attn_output)
        return output


class InteractionMapInit(nn.Module):
    """
    Version：
        药物分子先用EGNN得到坐标和结点特征,蛋白质还是GearNet得到结点特征(PocketDTA不变,相当于在预测阶段进行创新),
        通过各自结点特征以及之间的距离求和后作为pair的特征,pair特征计算是核心,
        类比CIGIN将编码器得到的特征(分子内特征)以及与pair矩阵相乘得到的特征(分之间特征)拼接后输入DNN得到结果,
        还能利用第一阶段的分子坐标作为辅助任务来约束EGNN的更新过程从而学习关键特征,这里就是直接计算多个回归loss,通过两个loss进行最终预测
        其中第一阶段的loss就是变化后药物分子的坐标与标签中坐标的差值之和RMSD(这里可能涉及到对药物分子坐标初始化的问题,因为蛋白质不动,
        药物分子初始化的坐标可以稍微远一点,但是不要有干涉问题),第二阶段的loss则是DTA的损失RMSE
        Parameters:
            hidden_dim: the hidden dim for lineared feature
            top_k: select the top k residue as the most important(pocket)
            scaled: scale the interaciton or not
    """
    def __init__(self, target_output_dim, drug_output_dim, hidden_dim=128, top_k=20, scaled=False):
        super(InteractionMapInit, self).__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.scaled = scaled

        self.linear_target = layers.MLP(target_output_dim, [hidden_dim], batch_norm=True)
        self.linear_drug = layers.MLP(drug_output_dim, [hidden_dim], batch_norm=True)
        self.linear_interaction = layers.MLP(hidden_dim, [1], batch_norm=True) # attention of single i,j node


    def forward(self, target_graph, drug_graph, target_feature, drug_feature):
        """
        输入包括第一阶段得到的特征以及各自的图结构，其中特征通过线性层得到一致大小 `hidden_dim`，
        这里想办法使用PackGraph进行运算，因为只能在每个DT-Pair内部计算interaction map，其他位置为0
        Parameters:
            target_graph: target protein graph
            drug_graph: drug molecule graph
            target_feature: [N_residue, hidden_dim]
            drug_feature: [N_atom, hidden_dim]
        """
        # Step 1. interaction map init with residue_feature i and atom_feature j [Num_residue(1), Num_atom(1), hidden_dim]
        target_feature = self.linear_target(target_feature["node_feature"])  # [Num_residue, hidden_dim]
        drug_feature = self.linear_drug(drug_feature["node_feature"])  # [Num_atom, hidden_dim]
        interaction_map = target_feature.unsqueeze(dim=1) - drug_feature.unsqueeze(dim=0)  # add by broadcast
        # TODO 注意药物的坐标应该会更新(EGNN)！仅固定蛋白质坐标

        # Step 2. interaction map add normed distance between i(residue) and j(atom)
        distance = (target_graph.node_position.unsqueeze(dim=1) - drug_graph.node_position.unsqueeze(dim=0)).norm(dim=-1)  # add by broadcast
        # packed_druggraph = data.Graph.from_dense(drug_graph)  # drug_graph.adjacency.to_dense().shape [152, 152, 4] 4 relations
        # TODO create a new function in this class for distance and interaction map update
        current_residue = 0  # record the i form [0, Num_residue]
        current_atom = 0  # record the j form [0, Num_atom]
        for index in range(target_graph.batch_size):
            # in a batch there are many DT-pair, residue_offset means the current protein residues, atom_offset is the same
            residue_offset = target_graph.num_residues[index]
            atom_offset = drug_graph.num_nodes[index]
            # setting the non-pair area in 0, like a dig metrix
            if index != target_graph.batch_size: # the last one need not to set 0
                interaction_map[current_residue:current_residue+residue_offset, current_atom+atom_offset:] = 0
                interaction_map[current_residue+residue_offset:, current_atom:current_atom+atom_offset] = 0
                distance[current_residue:current_residue+residue_offset, current_atom+atom_offset:] = 0
                distance[current_residue+residue_offset:, current_atom:current_atom+atom_offset] = 0
            # get the min and max in current DT-pair
            min_distance = torch.min(distance[current_residue:current_residue+residue_offset, current_atom:current_atom+atom_offset])
            max_distance = torch.max(distance[current_residue:current_residue+residue_offset, current_atom:current_atom+atom_offset])
            # update the distance in current DT-pair
            distance[current_residue:current_residue+residue_offset, current_atom:current_atom+atom_offset] =  (distance[current_residue:current_residue+residue_offset, current_atom:current_atom+atom_offset]- min_distance) / (max_distance - min_distance)
            current_residue += residue_offset  # update the starting index of residues
            current_atom += atom_offset  # update the starting index of atoms
        # TODO the distances should not be too close/far, publishment should be considered, the more close the more important
        interaction_map = interaction_map + distance.unsqueeze(dim=-1)  # [Num_residue, Num_atom, hidden_dim]
        # create a new function in this class for distance and interaction map update
        # Scale the map or not
        if self.scaled:
            interaction_map = interaction_map / (torch.sqrt(torch.tensor(self.hidden_dim)))
        interaction_map = torch.tanh(interaction_map)  # 500MB显存占用,后续换一下激活函数，进行实验看看哪种好
        # interaction_map = self.linear_interaction(interaction_map).squeeze(dim=-1)  # [Num_residue, Num_atom, hidden_dim]
        return interaction_map
        # Step 3. update the interaction map then get the top-k residue(the top-k max in the line i for example) for prediction(release the over-fitting probelm)
        # topk_residue = torch.mean(interaction_map, dim=-1)
        # _, topk_indices = torch.topk(topk_residue, k=self.top_k)  # 返回的是一个tuple(value, index)
        # return topk_indices # 用作pocket label的预测任务
        # 这里最好能创新一下，添加约束使筛选的残基最好空间上接近，因此上面添加的distance表示的药物与蛋白质之间的距离，这里还得加上内部距离 标准差 ，为了聚类到真实的pocket空间中
        # 同样的通过最大的几个残基就能预测出各自label来应对pocket预测子任务！
        # target_feature_after = torch.mm(interaction_map, drug_feature)
        # drug_feature_after = torch.mm(interaction_map.squeeze(dim=-1).t(), target_feature)
        # return target_feature_after, drug_feature_after


## 先用单层试一试效果，后期在设计叠加的网络，每一层里有2到3个模块，利用几何构型的注意力方法
class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11(True) and 12(Flase).
    interaction_map: [Num_residue, Num_atom, hidden_dim]
    Parameters:
        input_dim: the input dim for interaction map, like 128
        hidden_dim: the hidden dim for update the interaction map, like 128
        outgoing: use the outgoing or incoming edge for update, default `True`
    """
    def __init__(self, input_dim, hidden_dim, outgoing=True):
        """
        Args:
            interaction_dim: Input channel dimension
            hidden_dim: Hidden channel dimension
            outgoing: from the outgoing side or not
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.outgoing = outgoing

        # projection linear and gate linear for edge between a and b
        self.linear_ap = nn.Linear(input_dim, hidden_dim)
        self.linear_ag = nn.Linear(input_dim, hidden_dim)

        self.linear_bp = nn.Linear(input_dim, hidden_dim)
        self.linear_bg = nn.Linear(input_dim, hidden_dim)

        self.linear_g = nn.Linear(input_dim, input_dim)
        self.linear_z = nn.Linear(hidden_dim, input_dim)

        self.layernorm_in = nn.LayerNorm(input_dim)
        self.layernorm_out = nn.LayerNorm(hidden_dim)

        self.sigmoid = nn.Sigmoid()


    def combine_projections(self, a, b):
        if self.outgoing:
            # [hidden_dim, N_res, N_res]
            p = torch.matmul(
                a.permute(2, 0, 1), # [hidden_dim, N_res, N_atom]
                b.permute(2, 1, 0) # [hidden_dim, N_atom, N_res]
            )
        else:
            # [hidden_dim, N_atom, N_atom]
            p = torch.matmul(
                a.permute(2, 1, 0), # [hidden_dim, N_atom, N_res]
                b.permute(2, 0, 1), # [hidden_dim, N_res, N_atom]
            )
        # [N_res, N_res, hidden_dim] for outgoing
        # [N_atom, N_atom, hidden_dim] for incoming
        return p.permute(1, 2, 0)

    def forward(self, interaction_embedding):
        """
        Args:
            interaction_embedding: [Num_residue, Num_atom, hidden_dim] interaction map input
        Returns:
            [N_res, N_atom, interaction_dim] update the interaction embedding
        """
        interaction_embedding = self.layernorm_in(interaction_embedding)
        # [N_res, N_atom, hidden_dim]
        a = self.linear_ap(interaction_embedding) * self.sigmoid(self.linear_ag(interaction_embedding))
        # [N_res, N_atom, hidden_dim]
        b = self.linear_bp(interaction_embedding) * self.sigmoid(self.linear_bg(interaction_embedding))
        # the core difference is in or out
        x = self.combine_projections(a, b)  # [N_res, N_res, hidden_dim]
        x = self.layernorm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(interaction_embedding))  # [N_res, N_atom, hidden_dim]
        if self.outgoing:
            z = torch.matmul(x.permute(2, 0, 1), g.permute(2, 0, 1))  # [hidden_dim, N_res, N_atom]
        else:
            z = torch.matmul(g.permute(2, 0, 1), x.permute(2, 0, 1))
        return z.permute(1, 2, 0)  # [N_res, N_atom, hidden_dim]


# Custom loss for interaction map prediction
class Masked_BCELoss(nn.Module):
	def __init__(self):
		super(Masked_BCELoss, self).__init__()
		self.criterion = nn.BCELoss(reduce=False)

	def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
		batch_size = pred.size(0)
		loss_all = self.criterion(pred, label)
		loss_mask = torch.matmul(vertex_mask.view(batch_size,-1,1), seq_mask.view(batch_size,1,-1))*pairwise_mask.view(-1, 1, 1)
		loss = torch.sum(loss_all*loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
		return loss


class DeepFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DeepFusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, input):
        score = self.project(input)
        weight = torch.softmax(score, dim=1) # go for softmax in column
        return (weight * input).sum(dim=1)