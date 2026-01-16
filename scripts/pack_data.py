import json
import os
from glob import glob
from typing import List

import random
import tyro
from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from tqdm import tqdm

from giga_models.models.diffusion.giga_world_0 import T5TextEncoder
from giga_models.utils import download_from_huggingface


def pack_data_lerobot(
    video_dir: str,
    video_key: str,
    save_dir: str,
    text_encoder_model_path: str | None = None,
    device: str = 'cuda',
    sample_only: int | None = None,
    generate_train_val_test_splits: bool = False,
):
    """Pack LeRobot videos, prompts, and prompt embeddings into a dataset for training
    or evaluation.

    Args:
        data_dir: Directory containing .mp4 videos and corresponding .txt prompt files.
        save_dir: Directory to save the packed dataset.
        text_encoder_model_path: Path to T5 text encoder (download if None).
        device: Device for text encoder.
    """
    if text_encoder_model_path is None:
        text_encoder_model_path = download_from_huggingface('google-t5/t5-11b')
    # Load the T5 text encoder
    text_encoder = T5TextEncoder(text_encoder_model_path)
    text_encoder.to(device)

    os.makedirs(save_dir, exist_ok=True)

    # Find all video files
    video_paths = sorted(glob(os.path.join(video_dir, 'videos', 'chunk-*', video_key, '*.mp4')))
    if sample_only is not None:
        video_paths = video_paths[:sample_only]

    if generate_train_val_test_splits:
        num_videos = len(video_paths)
        shuffled_indices = list(range(num_videos))
        random.shuffle(shuffled_indices)
        train_indices = shuffled_indices[:int(0.8 * num_videos)]
        val_indices = shuffled_indices[int(0.8 * num_videos):int(0.9 * num_videos)]
        test_indices = shuffled_indices[int(0.9 * num_videos):]

        with open(os.path.join(save_dir, 'split_indices.json'), 'w') as f:
            json.dump({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }, f)

        splits = {
            'train': {'video_paths': [video_paths[idx] for idx in train_indices],
                      'save_dir': os.path.join(save_dir, 'train')},
            'val': {'video_paths': [video_paths[idx] for idx in val_indices],
                    'save_dir': os.path.join(save_dir, 'val')},
            'test': {'video_paths': [video_paths[idx] for idx in test_indices],
                     'save_dir': os.path.join(save_dir, 'test')},
        }

        print(f'Train videos: {len(splits["train"]["video_paths"])}')
        print(f'Val videos: {len(splits["val"]["video_paths"])}')
        print(f'Test videos: {len(splits["test"]["video_paths"])}')
        print('Split indices (train):', train_indices)
        print('Split indices (val):', val_indices)
        print('Split indices (test):', test_indices)
    else:
        splits = {'all': {'video_paths': video_paths, 'save_dir': save_dir}}


    ann_file = os.path.join(video_dir, 'meta', 'episodes.jsonl')
    if not os.path.exists(ann_file):
        print(f'Annotation file {ann_file} does not exist.')

    prompts = {}
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompts[record["episode_index"]] = " ".join(record["tasks"])

    for split_name, split_info in splits.items():
        video_paths_split = split_info['video_paths']
        save_dir_split = split_info['save_dir']
        os.makedirs(save_dir_split, exist_ok=True)
        print(f'Processing split: {split_name}, number of videos: {len(video_paths_split)}')

        # Writers for labels, videos, and prompt embeddings
        label_writer = PklWriter(os.path.join(save_dir_split , 'labels'))
        video_writer = FileWriter(os.path.join(save_dir_split, 'videos'))
        prompt_writer = FileWriter(os.path.join(save_dir_split, 'prompts'))

        for idx in tqdm(range(len(video_paths_split))):
            video_path = video_paths_split[idx]
            episode_index = int(os.path.basename(video_path).split('_')[1].split('.')[0])
            prompt = prompts[episode_index]

            # Encode the prompt to get embeddings
            prompt_embeds = text_encoder.encode_prompts(prompt)[0].cpu()

            label_dict = dict(data_index=idx, prompt=prompt)
            label_writer.write_dict(label_dict)
            video_writer.write_video(idx, video_paths_split[idx])
            prompt_writer.write_dict(idx, dict(prompt_embeds=prompt_embeds))

        # Finalize and close writers
        label_writer.write_config()
        video_writer.write_config()
        prompt_writer.write_config()
        label_writer.close()
        video_writer.close()
        prompt_writer.close()

        # Load datasets and combine into a single Dataset object
        label_dataset = load_dataset(os.path.join(save_dir_split, 'labels'))
        video_dataset = load_dataset(os.path.join(save_dir_split, 'videos'))
        prompt_dataset = load_dataset(os.path.join(save_dir_split, 'prompts'))
        dataset = Dataset([label_dataset, video_dataset, prompt_dataset])
        dataset.save(save_dir_split)
    


def pack_data(
    video_dir: str,
    save_dir: str,
    text_encoder_model_path: str | None = None,
    device: str = 'cuda',
):
    """Pack videos, prompts, and prompt embeddings into a dataset for training
    or evaluation.

    Args:
        video_dir: Directory containing .mp4 videos and corresponding .txt prompt files.
        save_dir: Directory to save the packed dataset.
        text_encoder_model_path: Path to T5 text encoder (download if None).
        device: Device for text encoder.
    """
    if text_encoder_model_path is None:
        text_encoder_model_path = download_from_huggingface('google-t5/t5-11b')
    # Load the T5 text encoder
    text_encoder = T5TextEncoder(text_encoder_model_path)
    text_encoder.to(device)
    # Find all video files
    video_paths: List[str] = glob(os.path.join(video_dir, '*.mp4'))
    # Writers for labels, videos, and prompt embeddings
    label_writer = PklWriter(os.path.join(save_dir, 'labels'))
    video_writer = FileWriter(os.path.join(save_dir, 'videos'))
    prompt_writer = FileWriter(os.path.join(save_dir, 'prompts'))
    for idx in tqdm(range(len(video_paths))):
        # For each video, read the corresponding prompt
        anno_file = video_paths[idx].replace('.mp4', '.txt')
        prompt = open(anno_file, 'r').read().strip()
        # Encode the prompt to get embeddings
        prompt_embeds = text_encoder.encode_prompts(prompt)[0].cpu()
        label_dict = dict(data_index=idx, prompt=prompt)
        label_writer.write_dict(label_dict)
        video_writer.write_video(idx, video_paths[idx])
        prompt_writer.write_dict(idx, dict(prompt_embeds=prompt_embeds))
    # Finalize and close writers
    label_writer.write_config()
    video_writer.write_config()
    prompt_writer.write_config()
    label_writer.close()
    video_writer.close()
    prompt_writer.close()
    # Load datasets and combine into a single Dataset object
    label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
    video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
    prompt_dataset = load_dataset(os.path.join(save_dir, 'prompts'))
    dataset = Dataset([label_dataset, video_dataset, prompt_dataset])
    dataset.save(save_dir)


if __name__ == '__main__':
    tyro.cli(pack_data)
