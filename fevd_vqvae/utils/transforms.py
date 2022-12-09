import torch

class RandomizePixels:
    def __init__(self, pixel_distort_prob=0.005):
        self._pixel_distort_prob = pixel_distort_prob

    def __call__(self, frames):
        t, c, h, w = frames.shape
        pixel_p = torch.full((t, c, h, w), fill_value=self._pixel_distort_prob)
        pixel_distort_bools = torch.bernoulli(pixel_p).to(bool)   
        noise = torch.rand((t, c, h, w))
        frames[pixel_distort_bools] = noise[pixel_distort_bools]
        return frames