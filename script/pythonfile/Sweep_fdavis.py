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


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings("ignore")

# Step 1: define the sweep config
sweep_configuration = {
    'method': 'grid',
    'parameters':
    {
        'learning_rate': {'values': [1e-4]}, 
        'weight_decay': {'values': [1e-4]},
        'protein_readout': {'values': ["mean", "sum"]},
        'drug_readout': {'values': ["mean", "sum"]},
        'protein_concat': {'values': [True]},
        'protein_short': {'values': [True]},
        'protein_hidden_dim': {'values': [512]},
        'protein_layer': {'values': [4]},
        'drug_concat': {'values': [False]},
        'drug_short': {'values': [False]},
        'drug_hidden_dim': {'values': [256]}, 
        'drug_layer': {'values': [5]},
        'fold': {'values': [0]},
        'random_seed': {'values': [42]},
        'batch_size': {'values': [128]}
    }
}


# Step 2: initialize sweep by passing in config. And provide a project name.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='Filtered_Davis_Sweep') 
# sweep_id = "marinehdk/Filtered_Davis_Sweep/wvhve8mh"


def main():
    wandb.init()
    seed_torch(wandb.config.random_seed)
    # ============================== Dataset Loading ============================== #
    protein_view = transforms.ProteinView(view="residue", keys="graph1")
    dataset = datasets.Davis(
        protein_method="gearnetesm_pocket", drug_method="distanceMol", description='filter', transform=protein_view
    )

    train_set, valid_set, test_set = dataset.deepdta_split(fold=wandb.config.fold)  
    print(f"Train samples: {len(train_set)}, Valid samples: {len(valid_set)}, Test samples: {len(test_set)}")

    # ============================== Model Defining  ============================== #
    protein_model = models.GearNet(
        input_dim=1280, hidden_dims=[wandb.config.protein_hidden_dim]*wandb.config.protein_layer,
        num_relation=7, batch_norm=True, short_cut=wandb.config.protein_short,
        concat_hidden=wandb.config.protein_concat, readout=wandb.config.protein_readout
    )
    drug_model = models.RGCN(
        input_dim=67, hidden_dims=[wandb.config.drug_hidden_dim] * wandb.config.drug_layer,
        num_relation=4, edge_input_dim=19, batch_norm=True, short_cut=wandb.config.drug_short,
        concat_hidden=wandb.config.drug_concat, readout=wandb.config.drug_readout
    )

    # ============================== Task Defining ============================== #
    task = tasks.InteractionPrediction(
        model=protein_model, model2=drug_model, mode_type='MolFormer', task=dataset.tasks, criterion="mse",
        metric=("rmse", "c_index", "spearmanr"), normalization=True, num_mlp_layer=3
    )
    task.apply(initialize_weights)

    # ============================== Normal Lr setting ============================== #
    optimizer = torch.optim.AdamW(
        params=task.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay
    )

    solver = core.Engine(
        task, train_set, valid_set, test_set, optimizer, None,
        gpus=[0], batch_size=wandb.config.batch_size, logger="wandb"
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-7
    )

    whole_params = sum(p.numel() for p in solver.model.parameters())
    print(f'#The Whole Params: {whole_params}')  

    # ============================== Training Begin ============================== #
    early_stopping = core.EarlyStopping(patience=30)
    checkpoint = "../../result/model_pth/pocket_fd_sweep.pth"

    # valid performance each epoch
    for epoch in range(200):
        print(">>>>>>>>   Model' LR: ", optimizer.param_groups[0]['lr'])
        solver.train()
        metric = solver.evaluate("valid")['root mean squared error [affinity]']  
        scheduler.step(metrics=metric)
        # add early stopping
        early_stopping(val_loss=metric, solver=solver, path=checkpoint)
        if early_stopping.early_stop:
            print(">>>>>>>>   Early stopping   >>>>>>>>")
            break
    # after all epoches test the performance
    solver.load(checkpoint)
    solver.evaluate("test")


# start the sweep job
wandb.agent(sweep_id=sweep_id, function=main, count=4)






