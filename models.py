from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output



class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),  # 65 -> 33
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 33 -> 17
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 17 -> 9
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 9 -> 5
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ViTEncoder(nn.Module):
    def __init__(self, latent_dim=256, pretrained=True):
        super().__init__()
        # Load Vision Transformer from torchvision
        self.vit = vit_b_16(pretrained=pretrained)
        self.vit.heads = nn.Identity()  # Remove classification head
        self.fc = nn.Linear(768, latent_dim)  # Map ViT output to latent_dim

    def forward(self, x):
        # Ensure input has 3 channels
        if x.shape[1] == 6:  # If input has 6 channels
            x = x[:, :3, :, :]  # Take only the first 3 channels
        elif x.shape[1] == 1:  # If grayscale, repeat to 3 channels
            x = x.repeat(1, 3, 1, 1)

        # Resize input to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Pass through ViT
        x = self.vit(x)
        return self.fc(x)
    
class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2, pretrained_vit=True):
        super().__init__()
        self.encoder = ViTEncoder(latent_dim=latent_dim, pretrained=pretrained_vit)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.target_encoder = ViTEncoder(latent_dim=latent_dim, pretrained=False)
        self.repr_dim = latent_dim

        # Copy encoder weights to target encoder (EMA initialization)
        for param_t, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            param_t.data.copy_(param.data)
            param_t.requires_grad = False

    def update_target_encoder(self, momentum=0.996):
        for param_t, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            param_t.data = param_t.data * momentum + param.data * (1 - momentum)

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        curr_state = self.encoder(states[:, 0])  # Encode the first state
        predictions = [curr_state]

        for t in range(actions.shape[1]):
            curr_state = self.predictor(torch.cat([curr_state, actions[:, t]], dim=1))
            predictions.append(curr_state)

        return torch.stack(predictions, dim=1)

    def training_step(self, states, actions):
        B, T, C, H, W = states.shape

        with torch.no_grad():
            target_repr = self.target_encoder(states.reshape(-1, C, H, W))
            target_repr = target_repr.reshape(B, T, -1)

        curr_state = self.encoder(states[:, 0])
        predictions = [curr_state]

        for t in range(T - 1):
            curr_state = self.predictor(torch.cat([curr_state, actions[:, t]], dim=1))
            predictions.append(curr_state)

        pred_repr = torch.stack(predictions, dim=1)
        loss_pred = nn.functional.mse_loss(pred_repr, target_repr)

        # Regularization terms
        std_pred = torch.sqrt(pred_repr.var(dim=0) + 1e-4)
        variance_loss = torch.mean(nn.functional.relu(1 - std_pred))

        pred_flat = pred_repr.reshape(-1, self.repr_dim)
        pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        cov = (pred_centered.T @ pred_centered) / (pred_centered.shape[0] - 1)
        cov_loss = (cov - torch.eye(cov.shape[0], device=pred_flat.device)).pow(2).sum()

        total_loss = loss_pred + 0.1 * variance_loss + 0.01 * cov_loss

        return {"loss": total_loss, "pred_loss": loss_pred, "var_loss": variance_loss, "cov_loss": cov_loss}