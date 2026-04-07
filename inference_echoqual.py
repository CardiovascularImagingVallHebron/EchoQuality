import os

import argparse
import numpy as np
import pandas as pd
import torch
import torch.hub
import cv2

from src.dataset import EchoQualityDataset
from src.models import MultiTaskEchoQ
from src.utils import Task, evaluate_echoq, set_seed

def load_parameters(parameters_path):
    parameters = {}
    with open(parameters_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                parameters[key.strip()] = value.strip()
    return parameters

def main(model_dir, root_path):
    # Identify parameters file
    parameters_path = 'results/echoQual_inference/parameters.txt'

    assert os.path.exists(parameters_path), f"Parameters file not found at {parameters_path}"

    # Load parameters
    parameters = load_parameters(parameters_path)
    
    # Extract relevant parameters
    data_dir = parameters.get('data_dir', '')
    dims = int(parameters.get('dims', 224))
    clip_len = int(parameters.get('clip_len', 16))
    num_clips = 4
    sampling_rate = int(parameters.get('sampling_rate', 1))
    normalization = parameters.get('normalization', '')
    fc_dropout = float(parameters.get('fc_dropout', 0.0))
    seed = int(parameters.get('seed', 0))
    split = 'test'
    set_seed(seed)
    
    tasks = np.load(os.path.join('content/tasks_v5_qual.npy'), allow_pickle=True)
    
    task_inclusion = []
    
    taskfold = 'quality'

    task_inclusion.append('view')
    task_inclusion.append('complete-structures')
    task_inclusion.append('calidad')
    task_inclusion.append('combined-quality')

    tasks = [t for t in tasks if t.task_name in task_inclusion]

    # Load test dataset
    print('Preparing test dataset...')
    
    rows = []

    # typical video extensions (adjust if needed)
    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    for patient_id in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_id)
        
        # skip if not a directory
        if not os.path.isdir(patient_path):
            continue
        
        # walk through all subfolders inside each study
        for root, _, files in os.walk(patient_path):
            for file in files:
                # check if file is a video
                if file.lower().endswith(video_exts):
                    
                    video_path = os.path.join(root, file)
                    video_path = os.path.normpath(video_path)
                    
                    # count number of frames using OpenCV
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        n_frames = None
                    else:
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # append row with required fields
                    rows.append({
                        "Patient_ID": patient_id,
                        "FilePath": video_path,
                        "FileName": file,
                        "NumberOfFrames": n_frames,
                        "complete-structures": 0,
                        "view": 0,
                        "calidad": 0,
                        "combined-quality": 0
                    })

    # create dataframe from collected rows
    test_df = pd.DataFrame(rows)

    # optional: sort results for consistency
    test_df = test_df.sort_values(["Patient_ID", "FileName"]).reset_index(drop=True)

    # Visualize the distribution in each fold
    fold = 1
    
    # Filter the data to ensure only the correct patients are used
    fold_val_df = test_df
    
    # Check
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
    print(tasks)

    # Load model
    print("Loading model...", flush=True)
    
    backbone_path = os.path.join('results', 'echoQual_inference', 'backbone_structure.pt')
    
    try:
        print("Trying torch.hub.load for backbone...", flush=True)
        backbone = torch.hub.load(
            'CarDS-Yale/PanEcho',
            'PanEcho',
            force_reload=False,
            backbone_only=True,
            clip_len=clip_len
        )
        print("Loaded backbone from torch.hub.", flush=True)
        
    except Exception as e:
        print(f"torch.hub.load failed ({e}). Trying local backbone: {backbone_path}", flush=True)
        backbone = torch.load(backbone_path, map_location='cpu')
        print("Loaded backbone from local file.", flush=True)
    
    model = MultiTaskEchoQ(
        backbone=backbone,
        tasks=tasks,
        fc_dropout=fc_dropout 
    )

    filtered_state_dict = {}
    checkpoint_path = 'results/echoQual_inference/echoqual.pt'

    # Load the full checkpoint
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint_path, flush=True)
    filtered_state_dict = {}

    for k, v in chkpt['weights'].items():
        filtered_state_dict[k] = v
            
    # Load the filtered layers
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
        model_dir=model_dir,
        weights=None,
        amp=False,
        plot_history=False,
        taskname = taskfold,
        fold = fold
    )

if __name__ == '__main__':
    model_dir = 'results/echoQual_inference'
    root_path = r"\\NAS3_Z\all\BKP_PERE\BBDD_Datasets\TEST_ECHOAVC\VALVE_VIDS"
    
    main(model_dir, root_path)
