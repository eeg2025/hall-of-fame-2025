import pdb

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import butter, resample_poly

from scipy.signal import butter, filtfilt, sosfiltfilt

import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3840):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)].to(x.device)

class EEGTransformerFull(nn.Module):
    """
    Full PyTorch model including all signal preprocessing steps
    from the MATLAB pipeline (fixed, non-trainable).
    """
    def __init__(self,
                 orig_fs=100, target_fs=64,
                 num_channels=128,
                 interesting_channels=[126, 125, 48, 112, 67, 93, 10, 61, 39, 108],
                 d_model=16, nhead=1, num_layers=1,
                 dropout=0.3, dim_feedforward=32,
                 highpass_cutoff=0.5,
                 artifact_scale=1.0,
                 batch_window_sec=2):
        super().__init__()
        self.orig_fs = orig_fs
        self.target_fs = target_fs
        self.highpass_cutoff = highpass_cutoff
        self.artifact_scale = artifact_scale
        self.batch_window = int(batch_window_sec * target_fs)
        self.interesting_channels = torch.tensor(interesting_channels, dtype=torch.long)

        # --- High-pass Butterworth filter coefficients ---
        b, a = butter(6, highpass_cutoff / (orig_fs / 2), btype='high', analog=False)
        self.register_buffer('b_hp', torch.tensor(b, dtype=torch.float32))
        self.register_buffer('a_hp', torch.tensor(a, dtype=torch.float32))

        # --- EEG channel projection ---
        self.channel_proj = nn.Sequential(
            nn.Conv1d(in_channels=len(interesting_channels),
                      out_channels=d_model,
                      kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # --- Transformer ---
        self.ln_proj = nn.LayerNorm(d_model)
        self.dropout_proj = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # --- Output head ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 1)
        self.dropout_fc = nn.Dropout(dropout)

        # --- Weight init ---
        for m in self.channel_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.fc.weight)


    # ---------- Fixed processing blocks ----------
    def remove_cz_and_select_channels(self, x):
        x = x[:, :-1, :]  # remove Cz
        x = x[:, self.interesting_channels, :]  # select subset
        return x

    def common_average_reference(self, x):
        return x - x.mean(dim=1, keepdim=True)


    def highpass_filter(self, x, cutoff=0.5, fs=100, order=6):
        """
        Apply zero-phase high-pass filter (MATLAB filtfilt equivalent)
        to a PyTorch tensor of shape (batch, channels, time).

        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal (batch, channels, time)
        cutoff : float
            High-pass cutoff frequency (Hz)
        fs : float
            Sampling frequency (Hz)
        order : int
            Filter order (default = 6)
        """

        # Convert to numpy for SciPy filtering
        x_np = x.detach().cpu().numpy()

        sos = butter(order, cutoff / (fs / 2), btype='high', output='sos')

        # Allocate array
        filtered = np.zeros_like(x_np)

        # Filter each channel for each batch
        for b in range(x_np.shape[0]):
            for ch in range(x_np.shape[1]):
                filtered[b, ch, :] = sosfiltfilt(sos, x_np[b, ch, :])

        # Convert back to tensor, same dtype/device as input
        return torch.from_numpy(filtered).to(x.device, dtype=x.dtype)


    def resample(self, x):
        # Use scipy resample_poly for fixed resampling
        x_np = x.detach().cpu().numpy()
        y_np = resample_poly(x_np, self.target_fs, self.orig_fs, axis=2)
        return torch.from_numpy(y_np).to(x.device, dtype=x.dtype)

    def artifact_suppression(self, x):
        return torch.tanh(x / self.artifact_scale)

    def zscore_normalise(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def segment_batches(self, x):
        """
        Cut EEG into non-overlapping 2s windows.
        Input shape: (B, C, T)
        Output shape: (B*num_segments, C, batch_window)
        """
        B, C, T = x.shape
        n_segments = T // self.batch_window
        x = x[:, :, :n_segments*self.batch_window]
        x = x.view(B, C, n_segments, self.batch_window)
        x = x.permute(0, 2, 1, 3).reshape(B*n_segments, C, self.batch_window)
        return x


    # ---------- Main forward ----------
    def forward(self, x, events=None):
        """
        x: (batch, 128, time at 500 Hz)
        events: optional (start, stop) indices to trim
        """
        # 1. Remove Cz, select interesting channels
        x = self.remove_cz_and_select_channels(x)



        # 2. Common average reference
        x = self.common_average_reference(x)

        # 3. High-pass filtering

        x = self.highpass_filter(x)

        # 4. Resampling 500 Hz → 64 Hz
        x = self.resample(x)


        # 6. Artifact suppression
        x = self.artifact_suppression(x)


        ##### Segmentation will be already made in the eval set
        # 7. Segment into 2-second windows
        #x = self.segment_batches(x)

        # 8. Z-score normalization per channel
        x = self.zscore_normalise(x)

        # 9. Channel projection and transformer
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.ln_proj(x)
        x = self.dropout_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # 10. Pool and output
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.dropout_fc(x)
        predictions = self.fc(x)


        return predictions
