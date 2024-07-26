import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EvINRModel(nn.Module):
    def __init__(self, n_layers=3, d_hidden=512, d_neck=256, H=260, W=346, recon_colors=True):
        super().__init__()

        self.recon_colors = recon_colors
        self.d_output = H * W * 3 if recon_colors else H * W
        self.net = Siren(
            n_layers=n_layers, d_input=1, d_hidden=d_hidden, d_neck=d_neck, 
            d_output=self.d_output 
        )
        self.H, self.W = H, W
        
    def forward(self, timestamps):
        log_intensity_preds = self.net(timestamps)
        if self.recon_colors:
            log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 3)
        else:
            log_intensity_preds = log_intensity_preds.reshape(-1, self.H, self.W, 1)
        return log_intensity_preds
    
    def get_losses(self, log_intensity_preds, event_frames):
        # temporal supervision to solve the event generation equation
        event_frame_preds = log_intensity_preds[1:] - log_intensity_preds[0: -1]
        temperal_loss = F.mse_loss(event_frame_preds, event_frames[:-1])
        # spatial regularization to reduce noise
        x_grad = log_intensity_preds[:, 1: , :, :] - log_intensity_preds[:, 0: -1, :, :]
        y_grad = log_intensity_preds[:, :, 1: , :] - log_intensity_preds[:, :, 0: -1, :]
        spatial_loss = 0.05 * (
            x_grad.abs().mean() + y_grad.abs().mean() + event_frame_preds.abs().mean()
        )

        # loss term to keep the average intensity of each frame constant
        const_loss = 0.1 * torch.var(
            log_intensity_preds.reshape(log_intensity_preds.shape[0], -1).mean(dim=-1)
        )
        return (temperal_loss + spatial_loss + const_loss)
    
    def tonemapping(self, log_intensity_preds, gamma=0.6):
        intensity_preds = torch.exp(log_intensity_preds).detach()
        # Reinhard tone-mapping
        intensity_preds = (intensity_preds / (1 + intensity_preds)) ** (1 / gamma)
        intensity_preds = intensity_preds.clamp(0, 1)
        return intensity_preds

# Roughly copy from https://github.com/vsitzmann/siren
class Siren(nn.Module):
    def __init__(
        self, n_layers, d_input, d_hidden, d_neck, d_output
    ):
        super().__init__()
        self.siren_net = []
        self.siren_net.append(SineLayer(d_input, d_hidden, is_first=True)) 
        for i_layer in range(n_layers):
            self.siren_net.append(SineLayer(d_hidden, d_hidden))
            if i_layer == n_layers - 1:
                self.siren_net.append(SineLayer(d_hidden, d_neck))
        self.siren_net.append(SineLayer(d_neck, d_output, is_last=True))
        self.siren_net = nn.Sequential(*self.siren_net)
        
    def forward(self, x):
        out = self.siren_net(x) # [B, H*W]
        return out
    
class SineLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, is_last=False, omega_0=10
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        if self.is_first:
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
        else:
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.in_features) / self.omega_0,
                np.sqrt(6 / self.in_features) / self.omega_0,
            )
                
    def forward(self, input):
        if self.is_last:
            return self.omega_0 * self.linear(input)
        else:
            return torch.sin(self.omega_0 * self.linear(input))