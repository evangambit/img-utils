import io, json, math, os, random, sqlite3, shutil, time
pjn = os.path.join

import numpy as np
import PIL
from PIL import Image
from scipy import fft, stats
import matplotlib.pyplot as plt

import importlib.resources as pkg_resources

__all__ = [
  'ImageSet'
]

kNumHashes = 64
kPcaClip = 4
kHashIconSize = 16
hash_w = None
hash_b = None
pca_v = None
pca_d = None
powmat = None

# How many hash-bits to allocate to each PCA dimension.
hashes_per_dim = None

def init():
  global hash_w, hash_b
  global pca_v, pca_d
  global powmat, hashes_per_dim
  pca_v = np.load(io.BytesIO(pkg_resources.read_binary(__package__, 'hash.eigenvectors.npy')))
  pca_d = np.load(io.BytesIO(pkg_resources.read_binary(__package__, 'hash.eigenvalues.npy')))
  powmat = 1 << np.arange(kNumHashes, dtype=np.uint64)
  hashes_per_dim = np.zeros(kHashIconSize*kHashIconSize, dtype=np.uint16)
  for _ in range(kNumHashes):
    hashes_per_dim[(pca_d / (hashes_per_dim + 1)**3).argmax()] += 1

  j = 0
  hash_w = np.zeros((kHashIconSize * kHashIconSize, kNumHashes))
  hash_b = np.zeros(kNumHashes)
  for d, n in enumerate(hashes_per_dim):
    for i in range(n):
      hash_w[:,j] = pca_v[d]
      z = stats.norm.ppf((i + 1) / (n + 1))
      z *= math.sqrt(pca_d[d])
      hash_b[j] = z
      j += 1

def img_hash(img):
  assert isinstance(img, Image.Image), type(img)
  img = img.convert('L')

  img = img.resize((16, 16), Image.BILINEAR)
  img = np.array(img, dtype=np.float64)
  diff = (img - img.mean()).flatten()

  diff = diff @ hash_w - hash_b

  hash1 = (diff > 0.0).astype(np.uint64)

  # Same as hash1, but the pixel nearest to the
  # threshold is flipped
  hash2 = hash1.copy()
  i = np.abs(diff).argmin()
  hash2[i] = 1 - hash2[i]

  hash1 = int((hash1 * powmat).sum()) - (1<<63)
  hash2 = int((hash2 * powmat).sum()) - (1<<63)
  return hash1, hash2

def compress(img16):
  assert isinstance(img16, Image.Image), type(img)
  a = io.BytesIO()
  img16.save(a, format='jpeg')
  return a.getvalue()

def decompress(data):
  assert isinstance(data, bytes), type(data)
  return Image.open(io.BytesIO(data))

