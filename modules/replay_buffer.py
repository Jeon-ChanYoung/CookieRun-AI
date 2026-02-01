import numpy as np
import torch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, config):
        self.episode_boundaries = []
        self.index = 0
        self.config = config

        print("📂 Loading combined data...")
        data = np.load(self.config.npz_dataset_path, allow_pickle=True)

        self.states_buffer = data["states"]     
        self.actions_buffer = data["actions"]    
        self.episode_boundaries = data["boundaries"].tolist()
        self.index = len(self.states_buffer)

        for ep in self.episode_boundaries:
            ep['length'] = ep['end'] - ep['start'] + 1

        self.print_statistics()
        print(f"Loaded {self.index} frames from {len(self.episode_boundaries)} episodes")
        
    def __len__(self):
        return self.index

    def get_valid_start_indices(self, seq_len):
        valid_indices = []

        for episode in self.episode_boundaries:
            start = episode['start']
            end = episode['end']

            valid_start = start
            valid_end = end - seq_len + 1

            valid_indices.extend(range(valid_start, valid_end + 1))

        return np.array(valid_indices)

    def sample_random(self, batch_size, seq_len):
        valid_indices = self.get_valid_start_indices(seq_len)

        if len(valid_indices) == 0:
            raise ValueError(f"No valid sequences of length {seq_len} available")

        if len(valid_indices) < batch_size:
            print(f"Not enough valid indices {len(valid_indices)} < {batch_size}")
            batch_size = len(valid_indices)

        start_indices = np.random.choice(valid_indices, batch_size)
        start_indices = start_indices.reshape(-1, 1)

        offsets = np.arange(seq_len).reshape(1, -1)
        indices = start_indices + offsets

        states = torch.as_tensor(self.states_buffer[indices], device=self.config.device)
        states = states.permute(0, 1, 4, 2, 3).float() / 255.0

        actions = torch.as_tensor(self.actions_buffer[indices], device=self.config.device)
        actions = F.one_hot(actions.long(), num_classes=self.config.action_size).float()

        return states, actions

    def sample_balanced(self, batch_size, seq_len):
        samples_per_episode = max(1, batch_size // len(self.episode_boundaries))
        all_indices = []

        for episode in self.episode_boundaries:
            start = episode['start']
            end = episode['end']
            episode_length = episode['length']

            if episode_length < seq_len:
                continue

            valid_starts = np.arange(start, end - seq_len + 2)

            n_samples = min(samples_per_episode, len(valid_starts))
            selected = np.random.choice(valid_starts, n_samples, replace=False)
            all_indices.extend(selected)

        if len(all_indices) < batch_size:
            valid_indices = self.get_valid_start_indices(seq_len)
            additional = np.random.choice(valid_indices, batch_size - len(all_indices))
            all_indices.extend(additional)

        start_indices = np.array(all_indices[:batch_size]).reshape(-1, 1)
        offsets = np.arange(seq_len).reshape(1, -1)
        indices = start_indices + offsets

        states = torch.as_tensor(self.states_buffer[indices], device=self.config.device)
        states = states.permute(0, 1, 4, 2, 3).float() / 255.0

        actions = torch.as_tensor(self.actions_buffer[indices], device=self.config.device)
        actions = F.one_hot(actions.long(), num_classes=self.config.action_size).float()

        return states, actions

    def sample_mixed(self, batch_size, seq_len, step, balanced_ratio=20):
        if step % balanced_ratio == 0:
            return self.sample_balanced(batch_size, seq_len)
        else:
            return self.sample_random(batch_size, seq_len)

    def print_statistics(self):
        print("\n" + "="*50)
        print("Replay Buffer Statistics")
        print("="*50)
        print(f"Total frames: {len(self)}")
        print(f"Total episodes: {len(self.episode_boundaries)}")
        print(f"\nEpisode details:")
        for ep in self.episode_boundaries:
            print(f"  • {ep['name']:15s}: {ep['length']:5d} frames  [{ep['start']:5d} → {ep['end']:5d}]")
        print("="*50 + "\n")