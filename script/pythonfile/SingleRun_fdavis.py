import os
import random
import torch
import warnings
import wandb
import numpy as np
from torchdrug import datasets, transforms, models, tasks, core


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_weights(m):
    # Conv1d init
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # Linear init
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # BatchNorm1d init
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


seed_torch()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings("ignore")

# ============================== dataset loading ============================== #
protein_view = transforms.ProteinView(view="residue", keys="graph1")
dataset = datasets.Davis(
    protein_method="gearnetesm_pocket", drug_method="distanceMol", description='filter', transform=protein_view
)

train_set, valid_set, test_set = dataset.deepdta_split(fold=0)
print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}")

# ============================== model define as the pretraining model ============================== #
protein_model = models.GearNet(
    input_dim=21, hidden_dims=[512, 512, 512, 512], num_relation=7, edge_input_dim=59, num_angle_bin=8,
    short_cut=False, batch_norm=True, concat_hidden=False, readout='mean'
)

drug_model = models.RGCN(
    input_dim=67, hidden_dims=[256, 256, 256, 256, 256], num_relation=4,
    edge_input_dim=19, batch_norm=True, readout='mean'
)
# ============================== task define and other training prepare ============================== #
task = tasks.InteractionPrediction(
    model=protein_model, model2=drug_model, task=dataset.tasks,  mode_type="MolFormer",
    criterion="mse", metric=("rmse", "c_index", "spearmanr"), hidden_dim=256, num_mlp_layer=3
)

task.apply(initialize_weights)

# ============================== Normal Lr setting ============================== #
# wandb.init(project="Filtered_Davis_Experiments")  # Best Filtered Davis Filtered_Davis_Experiments
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = torch.optim.AdamW(params=task.parameters(), lr=learning_rate, weight_decay=weight_decay)

solver = core.Engine(
    task, train_set, valid_set, test_set, optimizer, None, gpus=[0], batch_size=32  # , logger="wandb"
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-7
)

whole_params = sum(p.numel() for p in task.parameters())
print(f'#The Whole Params: {whole_params}')

# ============================== Training Begin ============================== #
early_stopping = core.EarlyStopping(patience=30)
checkpoint = "../../result/model_pth/gearnet_fd_0630.pth"

# valid performance each epoch
for epoch in range(200):
    print(">>>>>>>>   Model' LR: ", optimizer.param_groups[0]['lr'])
    solver.train()
    metric = solver.evaluate("valid")['root mean squared error [affinity]']  
    scheduler.step(metrics=metric)  # for reduceLR
    # add early stopping
    early_stopping(val_loss=metric, solver=solver, path=checkpoint)
    if early_stopping.early_stop:
        print(">>>>>>>>   Early Stopping   >>>>>>>>")
        break
# after all epoches test the performance
solver.load(checkpoint)
solver.evaluate("test")
