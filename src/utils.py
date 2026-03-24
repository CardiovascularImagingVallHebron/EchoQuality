import os
import gc
import random
import time
import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import csv

class Task():
    """Echocardiography interpretation task object."""
    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean

def merge_task_dicts(d):
    merged_dict = {}

    # Iterate through each dictionary in the list
    for dictionary in d:
        # Iterate through keys in the dictionary
        for key, value in dictionary.items():
            if key in merged_dict:
                # Merge lists if key is present in both merged_dict and the current dictionary
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in merged_dict[key]:
                            merged_dict[key][sub_key] += sub_value
                        else:
                            merged_dict[key][sub_key] = sub_value
                else:
                    merged_dict[key] += value
            else:
                # Otherwise, assign the corresponding dictionary
                merged_dict[key] = value

    return merged_dict

def time_elapsed(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f'{hours:.0f}h:{minutes:.0f}m:{seconds:.0f}s'

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_echoq(model, tasks, loss_fxns, data_loader, split, history, model_dir, weights, amp=False, plot_history=False, taskname=None, fold=0):
    s = time.perf_counter()
    model.eval()
    selected_taskname = "calidad"

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.

    overall_losses = []
    # task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [], 'yhats_sample': [], 'fnames': [], 'path': [], 'invalid_batches': 0} for task in tasks}
    task_data = {task.task_name: {'losses': [], 'ys': [], 'yhats': [],  'fnames': [], 'path': [], 'invalid_batches': 0} for task in tasks}
    
    # Define el archivo CSV y escribe los encabezados
    now = datetime.datetime.now()
    csv_file = os.path.join(model_dir, f"inference_sample_predictions_{now.strftime('%Y%m%d_%H%M%S')}_{selected_taskname}_{fold}.csv")

    # Keep metadata columns, but only export the selected task
    headers = ["batch", "path", "clip_index", "quality"]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    with torch.no_grad():
        for b, batch in pbar:
            # if b <= 361:
            #     continue
            x = batch['x']
            fname = batch['fname']
            path = batch['path']
            out_dicts = []

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                # Forward pass         
                for clip in range(x.shape[2]):
                    out_dict = model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d))    
                    out_dicts.append(out_dict)
                    
                    # Keep metadata columns and only write the selected task prediction
                    row = [b, path, clip]

                    # Extraer valores de cada task en orden dinámico
                    for task in tasks:
                        task_name = task.task_name
                        if task_name != selected_taskname:
                            continue
                        task_type = task.task_type
                        output = out_dict[task_name].cpu().detach().numpy()

                        if task_type == "multi-class_classification":
                            row.append(output[0])  # Suponiendo que es un solo valor
                        else:
                            row.append(output[0][0])  # Extraer solo el primer valor si es regresión o binario

                    # Guardar la fila en el CSV
                    with open(csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
                # Forward pass

                out_dicts = [model(x[:, :, clip, :, :, :].cuda(memory_format=torch.channels_last_3d)) for clip in range(x.shape[2])]

                # Compute loss for each task
                losses = []
                for task in tasks:
                    task_name = task.task_name
                    yhat = torch.stack([out_dict[task.task_name] for out_dict in out_dicts], dim=0).mean(dim=0)
                    y = batch[task.task_name].cuda()
                    mask = batch[task.task_name+'_mask'].cuda()


                    # If batch contains *only* missing values for task, then skip loss computation
                    if mask.sum() == 0:
                        task_data[task.task_name]['invalid_batches'] += 1
                        continue

                    # Mask out missing values from loss computation
                    masked_yhat = torch.masked_select(yhat, mask).reshape(-1, yhat.shape[1])
                    masked_y = torch.masked_select(y, mask).reshape(-1, y.shape[1])
                    
                    # Collect (masked) true and predicted labels
                    if task.task_type == 'multi-class_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.softmax(dim=1).numpy(force=True))
                    elif task.task_type == 'binary_classification':
                        task_data[task.task_name]['yhats'].append(masked_yhat.sigmoid().numpy(force=True))
                    else:
                        task_data[task.task_name]['yhats'].append(masked_yhat.numpy(force=True))

                    task_data[task.task_name]['ys'].append(masked_y.numpy(force=True))

                    # Collect (masked) auxiliary information
                    masked_fname = [f for f, m in zip(fname, mask) if m]
                    masked_paths = [p for p, m in zip(path, mask) if m]
                    task_data[task.task_name]['fnames'].append(masked_fname)
                    task_data[task.task_name]['path'].append(masked_paths)

                    # Store batch-level predictions, fname, and path
                    task_data[task.task_name]['batch_preds'] = task_data[task.task_name].get('batch_preds', [])
                    task_data[task.task_name]['batch_fnames'] = task_data[task.task_name].get('batch_fnames', [])
                    task_data[task.task_name]['batch_paths'] = task_data[task.task_name].get('batch_paths', [])

                    task_data[task.task_name]['batch_preds'].append(masked_yhat.cpu().numpy(force=True))
                    task_data[task.task_name]['batch_fnames'].append(fname)
                    task_data[task.task_name]['batch_paths'].append(path)

                    # For CrossEntropyLoss, target must have shape (N,)
                    if task.task_type == 'multi-class_classification':
                        masked_y = masked_y.squeeze(1)

                    # Compute task loss
                    loss = loss_fxns[task.task_name](masked_yhat, masked_y)
                    # Scale down regression loss based on mean value
                    if task.task_type == 'regression':
                        loss /= task.mean
                    losses.append(loss)

                    # Keep track of task losses for each batch
                    task_data[task.task_name]['losses'].append(loss.item())
                    del loss

                # Compute overall loss
                if len(losses) == 0:
                    continue
                else:
                    loss = sum(losses) / len(losses)
                    del losses
                    
                    # Keep running sum of losses for each batch
                    running_loss += loss.item()
                overall_losses.append(loss.item())

            pbar.set_postfix({'loss': running_loss/(b+1)})  # this is now a rough estimate using 1/n_gpu of the data

        # Write combined batch predictions to CSV at the end of each epoch
        combined_data = {}
        for task in tasks:
            task_name = task.task_name
            task_preds = []
            task_fnames = []
            task_paths = []
            
            for fnames, paths, preds in zip(task_data[task_name]['batch_fnames'],
                                           task_data[task_name]['batch_paths'],
                                           task_data[task_name]['batch_preds']):
                for fname, path, pred in zip(fnames, paths, preds):
                    task_fnames.append(fname)
                    task_paths.append(path)
                    task_preds.append(pred.tolist() if isinstance(pred, np.ndarray) else pred)

            # Merge task predictions into a combined dictionary
            for i, (fname, path, pred) in enumerate(zip(task_fnames, task_paths, task_preds)):
                if task_name != selected_taskname:
                    continue
                if (path, fname) not in combined_data:
                    combined_data[(path, fname)] = {'path': path, 'fname': fname}
                combined_data[(path, fname)][task_name] = pred
                
        # Convert combined data to a pandas DataFrame and save as CSV
        df = pd.DataFrame(list(combined_data.values()))
        df.rename(columns={'path': 'path', 'fname': 'fname', selected_taskname: 'quality'}, inplace=True)
        
        now = datetime.datetime.now()

        output_path = os.path.join(model_dir, f"inference_predictions_{now.strftime('%Y%m%d_%H%M%S')}_{selected_taskname}_{fold}.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    # if is_main_process():
    e = time.perf_counter()
    print(f'EVALUATION TIME: {time_elapsed(e-s)}')

    

