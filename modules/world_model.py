import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Normal, Independent, OneHotCategoricalStraightThrough, kl_divergence
from .network import Encoder, Decoder, RecurrentModel, PriorNetwork, PosteriorNetwork

class WorldModel:
    def __init__(self, config):
        self.config = config

        self.encoder = Encoder(self.config).to(self.config.device)
        self.decoder = Decoder(self.config).to(self.config.device)
        self.recurrentModel = RecurrentModel(self.config).to(self.config.device)
        self.priorNetwork = PriorNetwork(self.config).to(self.config.device)
        self.posteriorNetwork = PosteriorNetwork(self.config).to(self.config.device)

        self.world_model_parameters = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.recurrentModel.parameters()) +
            list(self.priorNetwork.parameters()) +
            list(self.posteriorNetwork.parameters())
        )

        self.world_model_optimizer = optim.AdamW(
            self.world_model_parameters,
            lr=self.config.world_model_lr,
            weight_decay=self.config.world_model_weight_decay
        )

        self.scaler = GradScaler()

        self.load_checkpoint(self.config.model_path)
        print("✅ WorldModel parameters loaded from checkpoint.")

    def dynamic_learning(self, experiences):
        states, actions = experiences

        with autocast(): 
            encoded_states = self.encoder(states)

            previous_recurrent_state = torch.zeros(
                self.config.batch_size, 
                self.config.recurrent_size, 
                device=self.config.device
            )
            previous_latent_state = torch.zeros(
                self.config.batch_size, 
                self.config.latent_size, 
                device=self.config.device
            )

            recurrent_states = []
            priors_logits = []
            posteriors = []
            posteriors_logits = []

            for t in range(1, self.config.batch_length):
                recurrent_state = self.recurrentModel(
                    previous_recurrent_state,
                    previous_latent_state,
                    actions[:, t-1]
                )

                _, prior_logits = self.priorNetwork(recurrent_state)
                posterior, posterior_logits = self.posteriorNetwork(
                    recurrent_state, 
                    encoded_states[:, t]
                )

                recurrent_states.append(recurrent_state)
                priors_logits.append(prior_logits)
                posteriors.append(posterior)
                posteriors_logits.append(posterior_logits)

                previous_recurrent_state = recurrent_state
                previous_latent_state = posterior

            recurrent_states  = torch.stack(recurrent_states,  dim=1) # (B, T-1, recurrent_size)
            priors_logits     = torch.stack(priors_logits,     dim=1) # (B, T-1, latent_length, latent_classes)
            posteriors        = torch.stack(posteriors,        dim=1) # (B, T-1, latent_length * latent_classes)
            posteriors_logits = torch.stack(posteriors_logits, dim=1) # (B, T-1, latent_length, latent_classes)

            # (B, T-1, recurrent_size + latent_length * latent_classes)
            full_states = torch.cat((recurrent_states, posteriors), dim=-1)

            ############# compute loss #############

            # Reconstruction loss
            reconstruction_means = self.decoder(full_states)
            reconstruction_distribution = Independent(
                Normal(reconstruction_means, 1),
                len(self.config.observation_shape)
            )
            reconstruction_loss = -reconstruction_distribution.log_prob(states[:, 1:]).mean()

            # KL loss
            prior_distribution       = Independent(OneHotCategoricalStraightThrough(logits=priors_logits             ), 1)
            prior_distributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()    ), 1)
            posterior_distribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits         ), 1)
            posterior_distributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1)

            prior_loss     = kl_divergence(posterior_distributionSG, prior_distribution  )
            posterior_loss = kl_divergence(posterior_distribution  , prior_distributionSG)
            free_nats      = torch.full_like(prior_loss, 1) # free nats = 1

            prior_loss = self.config.beta_prior * torch.maximum(prior_loss, free_nats)
            posterior_loss = self.config.beta_posterior * torch.maximum(posterior_loss, free_nats)
            kl_loss = (prior_loss + posterior_loss).mean()

            loss = reconstruction_loss + kl_loss

        ############# backprop #############

        self.world_model_optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.world_model_optimizer)
        nn.utils.clip_grad_norm_(
            self.world_model_parameters, 
            self.config.gradient_clip, 
            self.config.gradient_norm_type
        )
        self.scaler.step(self.world_model_optimizer)
        self.scaler.update()

        return reconstruction_loss.item(), kl_loss.item()

    def save_model_params(self, episode, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.config.map_name}_ep{episode}.pth')
        
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'recurrent_model': self.recurrent_model.state_dict(),
            'prior_network': self.prior_network.state_dict(),
            'posterior_network': self.posterior_network.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict()
        }, save_path)
        
        print(f"💾 Model saved: {save_path}")

    def load_checkpoint(self, check_point_path):
        print(f"📁 Loading checkpoint: {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=self.config.device)

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNetwork.load_state_dict(checkpoint['priorNetwork'])
        self.posteriorNetwork.load_state_dict(checkpoint['posteriorNetwork'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        print("Checkpoint loaded successfully.")