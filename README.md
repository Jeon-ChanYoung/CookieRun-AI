# CookieRun-AI
Play in an environment where an AI learns the first stage of Cookie Run, “The Witch's Oven,” and generates the next screen in real time based on your input.

<br>

| Item | Detail |
|------|--------|
| **Observation Size** | 128×256 pixels |
| **Action Space** | 3 actions (None, Jump, Slide) |
| **Training Data** | 50 real gameplay videos (~48,000 frames) |

<br>

## Training Data Distribution

> **Total Frames: 47,704** (from 50 real gameplay videos)

| Action | Label | Frames | Ratio |
|:------:|-------|-------:|------:|
| 0 | None | 35,773 | 75.0% |
| 1 | Jump | 1,249 | 2.6% |
| 2 | Slide | 10,682 | 22.4% |

<br>
  
## Real
<img src="assets/real.gif" width="512"/>

<br>
  
## Fake (AI-generated)
<img src="assets/fake.gif" width="512"/>

<br>

## Loss 

### VQ-VAE Loss
<img src="assets/fsq_vqvae_recon.png" alt="recon" width="600">

```
Epoch [ 1/30] VQ-VAE loss: 1.179069  recon: 0.040051  p_l: 3.796725
Epoch [ 2/30] VQ-VAE loss: 0.634990  recon: 0.026097  p_l: 2.029643 
Epoch [ 3/30] VQ-VAE loss: 0.459326  recon: 0.020848  p_l: 1.461594 
Epoch [ 4/30] VQ-VAE loss: 0.412156  recon: 0.020268  p_l: 1.306292 
Epoch [ 5/30] VQ-VAE loss: 0.403967  recon: 0.020089  p_l: 1.279593 
...
Epoch [26/30] VQ-VAE loss: 0.229845  recon: 0.014724  p_l: 0.717070 
Epoch [27/30] VQ-VAE loss: 0.234596  recon: 0.014913  p_l: 0.732277 
Epoch [28/30] VQ-VAE loss: 0.242575  recon: 0.015040  p_l: 0.758450  
Epoch [29/30] VQ-VAE loss: 0.245375  recon: 0.014927  p_l: 0.768160 
Epoch [30/30] VQ-VAE loss: 0.231396  recon: 0.014662  p_l: 0.722447  
```

<br>

### RSSM Loss
<img src="assets/rssm_recon.png" alt="recon" width="600">

<br>

<img src="assets/rssm_kl.png" alt="kl" width="600">

<br>

<img src="assets/rssm_acc.png" alt="acc" width="600">

```
Epoch [  1/400] RSSM loss: 3.573033  recon: 3.463033  kl: 1.100000  acc: 0.1418  top5: 0.4047
Epoch [  2/400] RSSM loss: 3.530106  recon: 3.420106  kl: 1.100000  acc: 0.1437  top5: 0.4121
Epoch [  3/400] RSSM loss: 3.526842  recon: 3.416842  kl: 1.100000  acc: 0.1437  top5: 0.4110
Epoch [  4/400] RSSM loss: 3.500003  recon: 3.390003  kl: 1.100000  acc: 0.1476  top5: 0.4177
Epoch [  5/400] RSSM loss: 3.488622  recon: 3.378612  kl: 1.100107  acc: 0.1486  top5: 0.4210
...
Epoch [396/400] RSSM loss: 0.815932  recon: 0.597647  kl: 2.182859  acc: 0.7973  top5: 0.9679
Epoch [397/400] RSSM loss: 0.877138  recon: 0.614608  kl: 2.625297  acc: 0.7909  top5: 0.9660
Epoch [398/400] RSSM loss: 0.822521  recon: 0.594985  kl: 2.275353  acc: 0.7975  top5: 0.9681
Epoch [399/400] RSSM loss: 0.823704  recon: 0.600613  kl: 2.230909  acc: 0.7963  top5: 0.9675
Epoch [400/400] RSSM loss: 0.809256  recon: 0.591078  kl: 2.181778  acc: 0.7989  top5: 0.9684
```

<br>

## How to Run  
**1. Clone the repository and install dependencies:** 
```
git clone https://github.com/Jeon-ChanYoung/Cookie-Run-AI.git
pip install -r requirements.txt
```

<br>

**2. Setup Pre-trained Model:**  
Download the pre-trained weights (vqvae_ep30.pth, rssm_ep400.pth) from the Releases page and place them in the directory structure as follows:  
```
model_params/
    └── rssm_ep400.pth
    └── vqvae_ep30.pth
```
If model_params does not exist, create it.  

<br>

**3. Run the main.py(FastAPI-based)**  
```
python main.py
```

<br>

## Simulation  
<img src="assets/simulation.gif" width="600"/>

- ⬆️ Arrow Up: Jump
- ⬇️ Arrow Down: Slide
- 🔄 R Key: Reset  

Due to the limited size of the training dataset, the model does not cover all possible game scenarios. Some visual artifacts or blur may appear during unusual gameplay patterns (e.g., repeated jump inputs). Additionally, the dataset only contains footage from the first stage; subsequent stages are not included.  
