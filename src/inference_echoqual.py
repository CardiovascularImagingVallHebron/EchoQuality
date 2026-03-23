import os

import argparse
import numpy as np
import pandas as pd
import torch
import torch.hub

from dataset import EchoQualityDataset
from models import MultiTaskEchoQ
from utils import Task, evaluate_echoq, set_seed

def load_parameters(parameters_path):
    parameters = {}
    with open(parameters_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                parameters[key.strip()] = value.strip()
    return parameters

def main(args):
    # Identify parameters file
    parameters_path = os.path.join(args.model_dir_path, 'parameters.txt')
    assert os.path.exists(parameters_path), f"Parameters file not found at {parameters_path}"

    # Load parameters
    parameters = load_parameters(parameters_path)
    
    # Extract relevant parameters
    data_dir = parameters.get('data_dir', '')
    dims = int(parameters.get('dims', 224))
    clip_len = int(parameters.get('clip_len', 16))
    num_clips = 4 #int(parameters.get('num_clips', 20))
    sampling_rate = int(parameters.get('sampling_rate', 1))
    normalization = parameters.get('normalization', '')
    fc_dropout = float(parameters.get('fc_dropout', 0.0))
    seed = int(parameters.get('seed', 0))
    split = 'test'
    set_seed(seed)
    print(dims, clip_len, sampling_rate, normalization, seed, flush=True)
    
    tasks = np.load(os.path.join('content/tasks_v5_qual.npy'), allow_pickle=True)
    
    for t in tasks:
        print(t.task_name)
    
    task_inclusion = []
    
    if args.view:
        taskfold = 'view'
    if args.structures:
        taskfold = 'complete-structures'
    if args.calidad:
        taskfold = 'calidad'
    if args.foreshortening:
        taskfold = 'foreshortening'
    if args.calidad_reg:
        taskfold = 'calidad-reg'
    if args.combined_quality:
        taskfold = 'combined_quality'

    task_inclusion.append('view')
    task_inclusion.append('complete-structures')
    task_inclusion.append('calidad')
    task_inclusion.append('combined-quality')


    tasks = [t for t in tasks if t.task_name in task_inclusion]

    # Load test dataset
    print('Preparing test dataset...')
    
    test_df = pd.read_excel('content/20260227_mimics_iv_quality.xlsx') #COLOR
    
    test_df['FilePath'] = test_df['FilePath'].apply(os.path.normpath)
    
    test_df['FileName'] = test_df['Period'].astype(str) + '-' + test_df['FileName'].astype(str)

    # Visualizar la distribución en cada fold
    fold = 1
    
    # Filtrar los datos asegurando que solo se usan los pacientes correctos
    fold_val_df = test_df
    
    # Comprobación
    print(f"Fold {fold+1}:", flush=True)
    print(f" - Test Patients: {len(fold_val_df)} ", flush=True)
    print("-" * 60, flush=True)
    test_dataset = EchoQualityDataset(
        data_dir=data_dir,
        data_df=fold_val_df,
        tasks=tasks,
        split=split,
        dims=dims,
        clip_len=clip_len,
        sampling_rate=sampling_rate,
        num_clips=num_clips,
        augment=False,
        normalization=normalization
    )

    print(f"Test dataset size: {len(test_dataset)}", flush=True)

    # Create loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # Load model
    print("Loading model...", flush=True)
    model = MultiTaskEchoQ(
        backbone=torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', force_reload=False, backbone_only=True, clip_len=clip_len),
        tasks=tasks,
        fc_dropout=fc_dropout 
    )

    filtered_state_dict = {}
    checkpoint_path = os.path.join(args.model_dir_path, f'chkpt_best_acc_calidad.pt')



    # Cargar el checkpoint completo
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint_path, flush=True)
    filtered_state_dict = {}

    for k, v in chkpt['weights'].items():
        filtered_state_dict[k] = v
            
    # Cargar las capas filtradas
    model.load_state_dict(filtered_state_dict, strict=True)

    model = model.cuda()

    # Define loss functions
    loss_fxns = {}
    for task in tasks:
        if task.task_type == 'regression':
            loss_fxns[task.task_name] = torch.nn.MSELoss()
        elif task.task_type == 'binary_classification':
            loss_fxns[task.task_name] = torch.nn.BCEWithLogitsLoss()
        elif task.task_type == 'multi-class_classification':
            loss_fxns[task.task_name] = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    evaluate_echoq(
        model=model,
        tasks=tasks,
        loss_fxns=loss_fxns,
        data_loader=test_loader,
        split=split,
        history=None,
        model_dir=args.model_dir_path,
        weights=None,
        amp=False,
        plot_history=False,
        taskname = taskfold,
        fold = fold
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--view', action='store_true', help="Include AoCalcium-presence task")
    parser.add_argument('--structures', action='store_true', help="Include AoCalcium-score task")
    parser.add_argument('--foreshortening', action='store_true', help="Include Fibrosis-score task")
    parser.add_argument('--calidad', action='store_true', help="Include AoCalcium-volume-score task")
    parser.add_argument('--calidad_reg', action='store_true', help="Include Fibrocalcific-score task")
    parser.add_argument('--combined_quality', action='store_true', help="Include GMAortico task")

    args = parser.parse_args()

    print(args)

    main(args)