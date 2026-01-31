import flax.linen as nn
import jax.numpy as jnp
from xminigrid.core.constants import NUM_COLORS, NUM_TILES


# code from https://github.com/dunnolab/xland-minigrid/blob/main/training/nn.py
class Embedding_xland_map(nn.Module):
    emb_dim: int = 16

    @nn.compact
    def __call__(self, img):
        entity_emb = nn.Embed(NUM_TILES, self.emb_dim)
        color_emb = nn.Embed(NUM_COLORS, self.emb_dim)

        img_emb = jnp.concatenate(
            [
                entity_emb(img[..., 0]),
                color_emb(img[..., 1]),
            ],
            axis=-1,
        )
        return img_emb
