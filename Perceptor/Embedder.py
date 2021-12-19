from pytti import *
import pytti

import math
import torch
from torch import nn
from torch.nn import functional as F

import kornia.augmentation as K

PADDING_MODES = {'mirror':'reflect','smear':'replicate','wrap':'circular','black':'constant'}

def sinc(x):
  return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
  cond = torch.logical_and(-a < x, x < a)
  out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
  return out / out.sum()
  

def ramp(ratio, width):
  n = math.ceil(width / ratio + 1)
  out = torch.empty([n])
  cur = 0
  for i in range(out.shape[0]):
    out[i] = cur
    cur += ratio
  return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
  n, c, h, w = input.shape
  dh, dw = size

  input = input.view([n * c, 1, h, w])

  if dh < h:
    kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
    pad_h = (kernel_h.shape[0] - 1) // 2
    input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
    input = F.conv2d(input, kernel_h[None, None, :, None])

  if dw < w:
    kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
    pad_w = (kernel_w.shape[0] - 1) // 2
    input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
    input = F.conv2d(input, kernel_w[None, None, None, :])

  input = input.view([n, c, h, w])
  return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class HDMultiClipEmbedder(nn.Module):
  """
  Multi-CLIP embedder that uses cutouts to view images larger than 224x224.
  with code by Katherine Crowson (https://github.com/crowsonkb)
  and jbusted (https://twitter.com/jbusted1)
  and dribnet (https://github.com/dribnet)
  """
  def __init__(self, perceptors=None, cutn = 64, cut_pow = 1.0, padding = 0.25, border_mode = 'clamp', noise_fac = 0.1):
    super().__init__()
    if perceptors is None:
      perceptors = pytti.Perceptor.CLIP_PERCEPTORS
    self.cut_sizes = [p.visual.input_resolution for p in perceptors]
    self.cutn = cutn
    self.noise_fac = noise_fac
    self.augs = nn.Sequential(K.RandomHorizontalFlip(p=0.5),
                              K.RandomSharpness(0.3, p=0.4),
                              K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
                              K.RandomPerspective(0.2, p=0.4,),
                              K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
                              nn.Identity(),)
    self.input_axes  = ('n', 's', 'y', 'x')
    self.output_axes = ('c', 'n', 'i')
    self.perceptors = perceptors
    self.padding = padding
    self.cut_pow = cut_pow
    self.border_mode = border_mode
    

  def forward(self, diff_image, input = None, device = DEVICE):
    """
    diff_image: (DifferentiableImage) input image
    returns images embeds
    """
    perceptors=self.perceptors
    side_x, side_y = diff_image.image_shape
    if input is None:
      input = format_module(diff_image, self).to(device = device, memory_format = torch.channels_last)
    else:
      input = format_input(input, diff_image, self).to(device = device, memory_format = torch.channels_last)
    max_size = min(side_x, side_y)
    min_size = min(side_x, side_y, self.cut_sizes[0])
    cutouts = []
    offsets = []
    sizes   = []
    for _ in range(self.cutn):
        size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
        offsetx = torch.randint(0, side_x - size + 1, ())
        offsety = torch.randint(0, side_y - size + 1, ())
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutouts.append(resample(cutout, (self.cut_sizes[0], self.cut_sizes[0])))
        offsets.append(torch.as_tensor([[offsetx / side_x, offsety / side_y]]).to(device))
        sizes.append(torch.as_tensor([[size / side_x, size / side_y]]).to(device))
    cutouts = self.augs(torch.cat(cutouts, dim=0))
    offsets = torch.cat(offsets)
    sizes   = torch.cat(sizes)
    if self.noise_fac:
      facs = cutouts.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
      cutouts.add_(facs * torch.randn_like(cutouts))
    clip_in = normalize(cutouts)
    image_embeds = []
    all_offsets = []
    all_sizes = []
    image_embeds.append(perceptors[0].encode_image(clip_in).float().unsqueeze(0))
    all_offsets.append(offsets)
    all_sizes.append(sizes)
    return cat_with_pad(image_embeds), torch.stack(all_offsets), torch.stack(all_sizes)
