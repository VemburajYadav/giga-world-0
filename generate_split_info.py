import os

base_path = os.path.join('/raid/vemburaj.yadav/datasets/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/')
train_split_path = os.path.join(base_path, 'train')
val_split_path = os.path.join(base_path, 'val')
test_split_path = os.path.join(base_path, 'test')

test_split_files = os.listdir(test_split_path)
test_split_indices = [int(f.split('.')[0]) for f in test_split_files if f.endswith('.json')]

train_split_files = os.listdir(train_split_path)
train_split_indices = [int(f.split('.')[0]) for f in train_split_files if f.endswith('.json')]

val_split_files = os.listdir(val_split_path)
val_split_indices = [int(f.split('.')[0]) for f in val_split_files if f.endswith('.json')]

splits = {
    'train': train_split_indices,
    'val': val_split_indices,
    'test': test_split_indices
}

with open('split_indices.json', 'w') as f:
    import json
    json.dump(splits, f)

    