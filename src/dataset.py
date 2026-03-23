import cv2
import numpy as np
import torch

from torchvision import tv_tensors
from torchvision.transforms import v2


class EchoQualityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_df, tasks, split, dims=128, clip_len=16, sampling_rate=1, num_clips=4, augment=False, normalization=''):
        self.data_dir = data_dir
        self.data_df = data_df
        self.tasks = tasks
        self.split = split
        self.clip_len = clip_len
        self.sampling_rate = sampling_rate
        self.num_clips = num_clips
        self.augment = augment
        self.normalization = normalization
        self.dims = dims
        
        # Set mean of each task (for current split)
        for task in self.tasks:
            task.mean = np.nanmean(self.data_df[task.task_name].values)

        print('---', split, '---')
        print(self.data_df)

        print(len(self.tasks))
        for task in self.tasks:
            print(vars(task))

        if self.normalization == 'imagenet':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'kinetics':  # kinetics-400
            self.mean = np.array([0.43216, 0.394666, 0.37645])
            self.std = np.array([0.22803, 0.22145, 0.216989])
        elif self.normalization == 'echo-clip':
            self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        else:
            self.mean = None
            self.std = None

        if self.augment:
            trsf_list = [
                v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5),
                v2.RandomCrop(size=(dims, dims)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
                v2.RandomRotation(degrees=(-30, 30)),
                v2.ToDtype(torch.float32, scale=True)
            ]
        elif self.split == 'tt-test' or self.split == 'tt-test-st':
            self.tt_augmentation_list = ['normal', 'zoom', 'rotate', 'brightness', 'offset', 'flip']
            trsf_list = []
        else:
            trsf_list = [
                v2.CenterCrop(size=(dims, dims)),
                v2.ToDtype(torch.float32, scale=True)
            ]

        if self.normalization != '':
            trsf_list += [v2.Normalize(mean=self.mean, std=self.std)]
            
        self.transform = v2.Compose(trsf_list)

    def _load_clip(self, fpath, frame_count, split, clip_len=16, sampling_rate=1, start_frame=None):
        """
        Load a clip from a video file with the given sampling rate.
        If the total frames are insufficient, reduce the sampling rate until it fits.
        """
        capture = cv2.VideoCapture(fpath)
        
        if not capture.isOpened():
            print(f"Error: Unable to open video file {fpath}") 

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure we can retrieve enough frames given the sampling rate
        while frame_count < clip_len * sampling_rate and sampling_rate > 1:
            sampling_rate -= 1

        try:
            if start_frame is None:
                start_idx = np.random.randint(0, frame_count)  # Random start index
            else:
                start_idx = start_frame
        except ValueError:
            print(f"ValueError for video {fpath} with frame_count {frame_count}")
            start_idx = 0

        v = []

        for i in range(clip_len):
            # Calculate the frame index, wrapping around if needed
            frame_idx = (start_idx + i * sampling_rate) % frame_count
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = capture.read()
            if not ret:
                # Imprime el nombre del archivo cuando hay un error al leer el frame
                print(f"Error leyendo frame {frame_idx} del video: {fpath}")
                # Handle unexpected read errors by appending a blank frame
                frame = np.zeros((self.dims, self.dims, 3), dtype=np.uint8)
            
            v.append(frame)

        capture.release()

        v = np.stack(v, axis=0)  # f x h x w x 3
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

        return v


    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx, :]  # info for a video

        fname = row['FileName']
        frame_count = row['NumberOfFrames']
        frame_path = row['FilePath']

        if self.split == 'train' or self.num_clips == 1:
            x = self._load_clip(frame_path, frame_count, self.split, self.clip_len)
            x = self.transform(x)
            x = torch.permute(x, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
            
        elif self.split == 'val' or self.split == 'test':
            x = []

            for i in range(self.num_clips):
                x_ = self._load_clip(frame_path, frame_count, self.split, self.clip_len)
                x_ = self.transform(x_)
                x_ = torch.permute(x_, (1, 0, 2, 3))  # f x 3 x h x w -> 3 x f x h x w
                x.append(x_)
            x = torch.stack(x, dim=1)

        elif self.split == 'tt-test' or self.split == 'tt-test-st':
            x = []
            zoom_levels = [0.8, 0.9, 1.2, 1.4]
            rotation_angles = [-15, -5, 5, 15]
            brightness_changes = [0.8, 0.9, 1.1, 1.2]
            crop_offsets = [(20, 20), (20, -20), (-20, -20), (-20, 20)]

            if self.split == 'tt-test':
                num_clips = self.num_clips
            else:
                segment_starts, num_clips, _ = calculate_segments(self.clip_len, self.sampling_rate, frame_count)
            
            for i in range(num_clips):
                if self.split == 'tt-test':
                    x_ = self._load_clip(frame_path, frame_count, self.split, self.clip_len)
                else:
                    x_ = self._load_clip(frame_path, frame_count, self.split, self.clip_len, self.sampling_rate, segment_starts[i])

                # Apply transformations sequentially
                if 'zoom' in self.tt_augmentation_list:
                    for zoom in zoom_levels:
                        transform = v2.Compose([
                            v2.RandomResizedCrop(size=(224, 224), scale=(zoom, zoom)),
                            v2.ToDtype(torch.float32, scale=True)
                        ])
                        x_zoomed = transform(x_)
                        x_zoomed = self.transform(x_zoomed)
                        x_zoomed = torch.permute(x_zoomed, (1, 0, 2, 3))
                        x.append(x_zoomed)

                if 'rotate' in self.tt_augmentation_list:
                    for angle in rotation_angles:
                        transform = v2.Compose([
                            v2.RandomRotation(degrees=(angle, angle)),
                            v2.CenterCrop(size=(224, 224)),
                            v2.ToDtype(torch.float32, scale=True)
                        ])
                        x_rotated = transform(x_)
                        x_rotated = self.transform(x_rotated)
                        x_rotated = torch.permute(x_rotated, (1, 0, 2, 3))
                        x.append(x_rotated)

                if 'brightness' in self.tt_augmentation_list:
                    for brightness in brightness_changes:
                        transform = v2.Compose([
                            v2.ColorJitter(brightness=brightness),
                            v2.CenterCrop(size=(224, 224)),
                            v2.ToDtype(torch.float32, scale=True)
                        ])
                        x_bright = transform(x_)
                        x_bright = self.transform(x_bright)
                        x_bright = torch.permute(x_bright, (1, 0, 2, 3))
                        x.append(x_bright)

                if 'offset' in self.tt_augmentation_list:
                    for offset in crop_offsets:
                        # Crop with offset
                        h, w = x_.shape[-2:]  # Height and width of the image
                        crop_h, crop_w = 224, 224

                        start_y = max(0, min((h - crop_h) // 2 + offset[0], h - crop_h))
                        start_x = max(0, min((w - crop_w) // 2 + offset[1], w - crop_w))

                        x_cropped = x_[..., start_y:start_y + crop_h, start_x:start_x + crop_w]
                        x_cropped = v2.Compose([
                            v2.ToDtype(torch.float32, scale=True)
                        ])(x_cropped)
                        x_cropped = self.transform(x_cropped)
                        x_cropped = torch.permute(x_cropped, (1, 0, 2, 3))
                        x.append(x_cropped)

                # Always apply flip
                if 'flip' in self.tt_augmentation_list:
                    transform = v2.Compose([
                        v2.CenterCrop(size=(224, 224)),
                        v2.RandomHorizontalFlip(p=1.0),
                        v2.ToDtype(torch.float32, scale=True)
                    ])
                    x_flipped = transform(x_)
                    x_flipped = self.transform(x_flipped)
                    x_flipped = torch.permute(x_flipped, (1, 0, 2, 3))
                    x.append(x_flipped)

                if 'normal' in self.tt_augmentation_list:
                    transform = v2.Compose([
                        v2.CenterCrop(size=(224, 224)),
                        v2.ToDtype(torch.float32, scale=True)
                    ])
                    x_ = transform(x_)
                    x_ = self.transform(x_)
                    x_ = torch.permute(x_, (1, 0, 2, 3))
                    x.append(x_)

            # Convert the list of transformed clips into a tensor
            x = torch.stack(x, dim=1)


        out_dict = {'x': x, 'fname': fname, 'path': frame_path}

        for task in self.tasks:
            is_nan = np.isnan(row[task.task_name])
            
            # if NAN, replace with -1 (will be masked later regardless)
            out_dict[task.task_name] = torch.LongTensor([row[task.task_name] if not is_nan else -1]) if task.task_type == 'multi-class_classification' else torch.FloatTensor([row[task.task_name] if not is_nan else -1])
            out_dict[task.task_name+'_mask'] = torch.BoolTensor([not is_nan])

        return out_dict

def calculate_segments(nb_frames, skipping_rate, total_frames):
    if total_frames < nb_frames * skipping_rate:
        while total_frames < nb_frames * skipping_rate:
            skipping_rate -= 1  # Reduce skipping_rate progresivamente

            if skipping_rate < 1:
                skipping_rate = 1
                break

    frame_range = nb_frames * skipping_rate
    segment_starts = []
    i = 0

    # Continuar mientras el último frame del último posible segmento no exceda el total de frames
    while i + frame_range <= total_frames:
        # Añadir índices de inicio para el rango actual
        for num in range(skipping_rate):
            if num + i + frame_range - skipping_rate <= total_frames :
                segment_starts.append(num + i)
                # segment_starts.extend(list(range(i, min(num + frame_range, total_frames - frame_range + 1))))
        # Saltar al próximo rango principal
        i += frame_range

    # Añadir índices adicionales al final si es necesario
    if total_frames - i > 15:  # Verificar si quedan más de 15 frames sin cubrir
        for j in range(3):
            start_index = total_frames - frame_range - j
            segment_starts.append(start_index)

    segment_starts = list(set(segment_starts))
    segment_starts.sort()
    total_segments = len(segment_starts)
    
    return segment_starts, total_segments, skipping_rate