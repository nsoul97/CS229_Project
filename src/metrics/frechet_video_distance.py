import torch
from typing import Tuple
import scipy
from typing import Any, List, Tuple, Union, Dict
import numpy as np
import os
import shutil
import sys
import types
import io
import re
import requests
import html
import hashlib
import urllib
import urllib.request
import torchvision.transforms.functional as F


class FVD:

    def __init__(self,
                 max_val: int,
                 normalize: bool = True,
                 target_resolution_dim: int = 224,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self._detector_kwargs = dict(rescale=False, resize=False,
                                     return_features=True)  # Return raw features before the softmax layer.

        with self._open_url(url=detector_url, verbose=False) as f:
            self._detector = torch.jit.load(f).eval().to(device)
        self._device = device
        self._frame_target_resolution = [target_resolution_dim] * 2
        self._max_val = max_val
        self._normalize = normalize

    def _open_url(self,
                  url: str,
                  num_attempts: int = 10,
                  verbose: bool = True,
                  return_filename: bool = False) -> Any:
        """Download the given URL and return a binary-mode file object to access the data."""
        assert num_attempts >= 1

        # Doesn't look like an URL scheme so interpret it as a local filename.
        if not re.match('^[a-z]+://', url):
            return url if return_filename else open(url, "rb")

        # Handle file URLs.  This code handles unusual file:// patterns that
        # arise on Windows:
        #
        # file:///c:/foo.txt
        #
        # which would translate to a local '/c:/foo.txt' filename that's
        # invalid.  Drop the forward slash for such pathnames.
        #
        # If you touch this code path, you should test it on both Linux and
        # Windows.
        #
        # Some internet resources suggest using urllib.request.url2pathname() but
        # but that converts forward slashes to backslashes and this causes
        # its own set of problems.
        if url.startswith('file://'):
            filename = urllib.parse.urlparse(url).path
            if re.match(r'^/[a-zA-Z]:', filename):
                filename = filename[1:]
            return filename if return_filename else open(filename, "rb")

        url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

        # Download.
        url_name = None
        url_data = None
        with requests.Session() as session:
            if verbose:
                print("Downloading %s ..." % url, end="", flush=True)
            for attempts_left in reversed(range(num_attempts)):
                try:
                    with session.get(url) as res:
                        res.raise_for_status()
                        if len(res.content) == 0:
                            raise IOError("No data received")

                        if len(res.content) < 8192:
                            content_str = res.content.decode("utf-8")
                            if "download_warning" in res.headers.get("Set-Cookie", ""):
                                links = [html.unescape(link) for link in content_str.split('"') if
                                         "export=download" in link]
                                if len(links) == 1:
                                    url = requests.compat.urljoin(url, links[0])
                                    raise IOError("Google Drive virus checker nag")
                            if "Google Drive - Quota exceeded" in content_str:
                                raise IOError("Google Drive download quota exceeded -- please try again later")

                        match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                        url_name = match[1] if match else url
                        url_data = res.content
                        if verbose:
                            print(" done")
                        break
                except KeyboardInterrupt:
                    raise
                except:
                    if not attempts_left:
                        if verbose:
                            print(" failed")
                        raise
                    if verbose:
                        print(".", end="", flush=True)

        # Return data as file object.
        assert not return_filename
        return io.BytesIO(url_data)

    def _features_fvd(self,
                      feats_fake: np.ndarray,
                      feats_real: np.ndarray) -> float:

        mu_gen, sigma_gen = self._compute_stats(feats_fake)
        mu_real, sigma_real = self._compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def _compute_stats(self,
                       feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]
        return mu, sigma

    def __call__(self,
                 videos_real: torch.Tensor,
                 videos_fake: torch.Tensor) -> float:
        """

        :param videos_fake: A tensor with shape [B, T, C, H, W] containing T frames of B fake videos
        :param videos_real: A tensor with shape [B, T, C, H, W] containing T frames of B real videos
        :return: The FVD for the given real and generated videos.
        """

        B, T, C, H, W = videos_fake.shape

        videos_fake = videos_fake.float().to(self._device)
        videos_real = videos_real.float().to(self._device)

        if H != self._frame_target_resolution or W != self._frame_target_resolution:
            frames_real = videos_real.reshape(-1, C, H, W)
            frames_fake = videos_fake.reshape(-1, C, H, W)

            videos_real = F.resize(frames_real, self._frame_target_resolution).reshape(B, T, C, *self._frame_target_resolution)
            videos_fake = F.resize(frames_fake, self._frame_target_resolution).reshape(B, T, C, *self._frame_target_resolution)

        videos_fake = videos_fake.transpose(1, 2)
        videos_real = videos_real.transpose(1, 2)

        videos_real = videos_real / self._max_val
        videos_fake = videos_fake / self._max_val

        if self._normalize:
            videos_real = 2. * videos_real - 1.
            videos_fake = 2. * videos_fake - 1.

        with torch.inference_mode():
            feats_fake = self._detector(videos_fake, **self._detector_kwargs).cpu().numpy()
            feats_real = self._detector(videos_real, **self._detector_kwargs).cpu().numpy()

        return self._features_fvd(feats_fake, feats_real)
