import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# for training RSSM
def precompute_vae_latents(config, vae, frame_dataset, batch_size=256):
    with torch.no_grad():
        vae.change_train_mode(train=False)
        loader = DataLoader(frame_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=2, pin_memory=True)
        all_z = []
        for frames in loader:
            frames = frames.to(config.device, non_blocking=True)
            all_z.append(vae.encode(frames).cpu())
        return torch.cat(all_z, dim=0)  # (N, 4, 16, 32)


def straight_through_categorical(logits):
    probs = F.softmax(logits, dim=-1)
    hard = F.one_hot(probs.argmax(dim=-1), logits.shape[-1]).float()
    return hard - probs.detach() + probs