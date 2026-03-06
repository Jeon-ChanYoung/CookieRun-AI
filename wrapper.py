import torch
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os
import cv2

class Wrapper:
    def __init__(self, config, vqvae, rssm):
        self.config = config
        self.vqvae = vqvae
        self.rssm = rssm

        self.action_map = {
            "none": 0,
            "jump": 1,
            "slide": 2,
        }

        self._load_samples()

        self._action_tensors = {
            name: self._create_action_tensor(idx)
            for name, idx in self.action_map.items()
        }

        self.reset()
        print("Game state initialized")

    def _create_action_tensor(self, action_idx):
        action = torch.zeros(1, self.config.action_size, device=self.config.device)
        action[0, action_idx] = 1.0
        return action

    @torch.no_grad()
    def reset(self):
        self.recurrent_state = torch.zeros(
            1, self.config.recurrent_size, device=self.config.device
        )

        initial_img = self.single_state_sample()         # (1, 3, 128, 256)
        initial_indices = self.vqvae.encode(initial_img) # (1, 16, 32) LongTensor

        encoded_state = self.rssm.encoder(initial_indices)    # (1, encoded_state_size)

        action = self._action_tensors["none"]

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            torch.zeros(1, self.config.latent_size, device=self.config.device),
            action,
        )

        self.latent_state, _ = self.rssm.representation_model(
            self.recurrent_state, encoded_state
        )

        return self.get_current_image()

    @torch.no_grad()
    def step(self, action_name: str):
        action_tensor = self._action_tensors.get(
            action_name, self._action_tensors["none"]
        )

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            self.latent_state,
            action_tensor,
        )

        self.latent_state, _ = self.rssm.transition_model(self.recurrent_state)

        return self.get_current_image()

    @torch.no_grad()
    def get_current_image(self):
        # RSSM decoder outputs VQ token logits: (1, K=256, 16, 32)
        token_logits = self.rssm.decoder(self.recurrent_state, self.latent_state)

        token_indices = token_logits.argmax(dim=-3)  # (1, 16, 32)

        # VQ-VAE decoder: indices -> pixel image (1, 3, 128, 256)
        reconstruction = self.vqvae.decode(token_indices)

        img = reconstruction[0].clamp(0, 1)
        img = (img * 255).byte().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        return img

    def image_to_base64(self, img):
        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format="WEBP", quality=90)

        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/webp;base64,{img_base64}"

    def single_state_sample(self):
        idx = np.random.randint(0, len(self.sample_images))
        img = self.sample_images[idx]  # (128, 256, 3) uint8

        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 256)
        img_tensor = img_tensor.to(self.config.device)

        return img_tensor

    def _load_samples(self):
        samples_dir = "samples/oven_of_witch"

        if not os.path.exists(samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

        self.sample_images = []

        for filename in sorted(os.listdir(samples_dir)):
            if not filename.startswith("oow_sample") or not filename.endswith(".png"):
                continue

            file_path = os.path.join(samples_dir, filename)
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.sample_images.append(img_rgb)

        print(f"✅ Loaded {len(self.sample_images)} sample images")