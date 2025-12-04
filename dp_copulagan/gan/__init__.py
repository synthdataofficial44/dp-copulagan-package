"""GAN architecture module for DP-CopulaGAN."""

from dp_copulagan.gan.spectral_norm import SpectralNormalization
from dp_copulagan.gan.generator import build_generator, build_generator_unconditional
from dp_copulagan.gan.critic import build_critic, build_critic_unconditional
from dp_copulagan.gan.gradient_penalty import gradient_penalty

__all__ = [
    'SpectralNormalization',
    'build_generator',
    'build_generator_unconditional',
    'build_critic',
    'build_critic_unconditional',
    'gradient_penalty',
]
