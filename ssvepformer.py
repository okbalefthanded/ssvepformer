# Okba Bekhelifi dec 2024, <okba.bekhelifi@univ-usto.dz>
# Implements SSVEPFormer model from:
# Chen, J. et al. (2023) ‘A transformer-based deep neural network model for SSVEP classification’, 
# Neural Networks, 164, pp. 521–534. Available at: https://doi.org/10.1016/j.neunet.2023.04.045.
#
#

from torch import flatten
from torch import nn
import torch.nn.functional as F

class ChComb(nn.Module):
  def __init__(self, Chans=8, Samples=220, dropout=0.5):
    super().__init__()
    self.conv = nn.Conv1d(Chans // 2, Chans, 1, padding='same')
    self.ln   = nn.LayerNorm(Samples)
    self.act  = nn.GELU()
    self.do   = nn.Dropout(p=dropout)

  def forward(self, x):
    return self.do(self.act(self.ln(self.conv(x))))

class Encoder(nn.Module):
  def __init__(self, Chans=16, Samples=220, dropout=0.5):
    super().__init__()
    # CNN module
    self.channels = Chans
    self.ln1  = nn.LayerNorm(Samples)
    self.conv = nn.Conv1d(Chans, Chans, 31, padding='same')
    self.ln2  = nn.LayerNorm(Samples)
    self.act  = nn.GELU()
    self.do   = nn.Dropout(p=dropout)
    # MLP module
    self.ln3  = nn.LayerNorm(Samples)
    self.proj = nn.Linear(Chans, Samples)
    self.do2  = nn.Dropout(p=dropout)

  def forward(self, x):
    #
    shortcut1 = x
    x = self.conv(self.ln1(x))
    x = self.act(self.ln2(x))
    x = self.do(x) + shortcut1
    shortcut2 = x
    #
    x = self.ln3(x)
    output_channels = []
    for i in range(self.channels):
      c = self.proj(x[:,:,i])
      c = c.unsqueeze(1)
      output_channels.append(c)
    x = torch.cat(output_channels, 1)
    x = self.do(x) + shortcut2
    return x

class MlpHead(nn.Module):
  def __init__(self, Chans, Samples, n_classes, drop_rate=0.5):
    super().__init__()
    self.drop       = nn.Dropout(drop_rate)
    self.linear1    = nn.Linear(Chans * Samples, 6 * n_classes)
    self.norm       = nn.LayerNorm(6*n_classes)
    self.activation = nn.GELU()
    self.drop2      = nn.Dropout(drop_rate)
    self.linear2    = nn.Linear(6*n_classes, n_classes)

  def forward(self, x):
    x = flatten(x, 1)
    x = self.drop(x)
    x = self.linear1(x)
    x = self.norm(x)
    x = self.activation(x)
    x = self.drop2(x)
    x = self.linear2(x)
    return x

class SSVEPFormerTH(nn.Module):
  def __init__(self, Chans=8, n_classes=12, fs=256,
               band=[8, 64], resolution=0.25, 
               drop_rate=0.25):
    super().__init__()
    self.name = "SSVEPFORMER"
    self.fs = fs
    self.resolution = resolution
    self.nfft  = round(fs / resolution)
    self.fft_start = int(round(band[0] / self.resolution))
    self.fft_end   = int(round(band[1] / self.resolution)) + 1
    samples = (self.fft_end - self.fft_start) * 2
    filters = 2*Chans

    self.channel_comb = ChComb(filters,  samples, drop_rate)
    self.encoder1     = Encoder(filters, samples, drop_rate)
    self.encoder2     = Encoder(filters, samples, drop_rate)
    self.head         = MlpHead(filters, samples, n_classes, drop_rate)

    self.init_weights()

  def init_weights(self):
    for module in self.modules():
        if hasattr(module, 'weight'):
          cls_name = module.__class__.__name__
          if not("BatchNorm" in cls_name or "LayerNorm" in cls_name):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
          else:
            nn.init.constant_(module.weight, 1)
          if hasattr(module, "bias"):
            if module.bias is not None:
              nn.init.constant_(module.bias, 0)

  def forward(self, x):
    x = self.transform(x)
    x = self.channel_comb(x)
    x = self.encoder1(x)
    x = self.encoder2(x)
    x = self.head(x)
    return x

  def transform(self, x):
    with torch.no_grad():
      samples = x.shape[-1]
      x = torch.fft.fft(x, n=self.nfft) / samples
      real = x.real[:,:, self.fft_start:self.fft_end]
      imag = x.imag[:,:, self.fft_start:self.fft_end]
      x = torch.cat((real, imag), axis=-1)
    return x


class FBSSVEPFormer(nn.Module):
  def __init__(self, fs=256, n_subbands=3, models=None):
    super().__init__()
    self.name = "FB-SSVEPFORMER"
    self.fs = fs
    self.subbands = [[8*i, 80] for i in range(1, n_subbands+1)]
    self.subnets  = models
    self.conv     = nn.Conv1d(n_subbands, 1, 1, padding='same')
    self.init_weights()

  def init_weights(self):
    nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
    nn.init.constant_(self.conv.bias, 0)

  def forward(self, x):
    out = []
    for i, band in enumerate(self.subbands):
      c = self.filter_band(x, band)
      c = self.subnets[i](c)
      c = c.unsqueeze(1)
      out.append(c)
    #
    x = torch.cat(out, 1)
    x = self.conv(x)
    return x.squeeze(1)

  def filter_band(self, x, band):
    # x: batch, channels, samples
    device = x.device
    with torch.no_grad():
      x = x.cpu().numpy()
      B, A = butter(4, np.array(band) / (self.fs / 2), btype='bandpass')
      x = filtfilt(B, A, x, axis=-1)
      x = x.copy()
    return torch.tensor(x, dtype=torch.float, device=device)

