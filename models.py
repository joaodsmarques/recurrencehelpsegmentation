import segmentation_models_pytorch as smp
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F


class CRFBlock(nn.Module):
    # One CRF-as-RNN step: pixels exchange label beliefs with k x k neighbours
    # that look similar in the image.

    def __init__(self, num_classes=2, kernel_size=7, downsample=2, normalize=False):
        super().__init__()
        self.k = kernel_size
        self.downsample = (
            downsample  # 4 → CRF runs on a 128x128 grid for 512x512 inputs
        )
        self.normalize = normalize  # neighbour weights sum to 1
        self.log_theta_color = nn.Parameter(torch.tensor(0.0))
        self.log_theta_space = nn.Parameter(torch.tensor(1.0))
        self.compat = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        nn.init.zeros_(self.compat.weight)  # starts identical to the base model

    def forward(self, logits, unary, image):
        H, W = logits.shape[-2:]
        if self.downsample > 1:
            image = F.avg_pool2d(
                image, self.downsample
            )  # also averages out pixel noise
            logits = F.avg_pool2d(logits, self.downsample)
        B, L, h, w = logits.shape
        k, p, K = self.k, self.k // 2, self.k**2

        r = torch.arange(k, device=image.device, dtype=image.dtype) - p
        space = (r.view(-1, 1) ** 2 + r.view(1, -1) ** 2).view(1, K, 1, 1)
        img = F.unfold(image, k, padding=p).view(B, image.shape[1], K, h, w)
        color = ((img - image.unsqueeze(2)) ** 2).sum(1)
        kernel = torch.exp(
            -space / (2 * self.log_theta_space.exp() ** 2)
            - color / (2 * self.log_theta_color.exp() ** 2)
        )
        self_mask = torch.ones(1, K, 1, 1, device=image.device, dtype=image.dtype)
        self_mask[:, K // 2] = 0
        kernel = kernel * self_mask  # no message to itself
        if self.normalize:
            kernel = kernel / (kernel.sum(1, keepdim=True) + 1e-6)

        q = F.unfold(F.softmax(logits, 1), k, padding=p).view(B, L, K, h, w)
        pairwise = self.compat((q * kernel.unsqueeze(1)).sum(2))
        if self.downsample > 1:
            pairwise = F.interpolate(
                pairwise, size=(H, W), mode="bilinear", align_corners=False
            )
        return unary - pairwise


class HopfieldBlock(nn.Module):
    """One modern-Hopfield retrieval step: patches of the logit map retrieve
    learned memory patterns."""

    # Checkpoints trained before 2026-10-29 used patch_size=32, stride=16 and log_beta 2.0:
    # smaller patches (512-dim instead of 2048-dim) and a larger beta make the retrieval less
    # flat (with 2048 dims, cosine scores to random memories are all ~0)
    def __init__(
        self,
        num_classes=2,
        patch_size=16,
        n_memories=512,
        normalize=True,
        stride=8,
        use_probs=False,
    ):
        super().__init__()
        self.p = patch_size
        self.stride = stride or patch_size  # patch_size // 2 → overlapping patches
        self.normalize = normalize  # cosine similarity instead of raw dot product
        # Mean-field style (like CRFBlock): retrieve from softmax(logits) instead of the logits,
        # so a correction at uncertain pixels changes the next step's query
        self.use_probs = use_probs
        dim = patch_size * patch_size * num_classes
        self.memories = nn.Parameter(torch.randn(n_memories, dim) / dim**0.5)
        # retrieval sharpness; cosine scores lie in [-1, 1], so they need a larger beta
        self.log_beta = nn.Parameter(torch.tensor(3.0 if normalize else 0.0))
        self.gate = nn.Parameter(
            torch.tensor(0.0)
        )  # starts identical to the base model

    def forward(self, logits, unary, image):
        B, L, H, W = logits.shape
        # Probabilities are centred (q - 1/L): all-positive patches would otherwise all
        # look alike under cosine similarity; uncertain pixels are ~0, confident ones ±
        query = F.softmax(logits, 1) - 1.0 / L if self.use_probs else logits
        tokens = F.unfold(query, self.p, stride=self.stride).transpose(
            1, 2
        )  # B, N, dim
        keys = self.memories
        if self.normalize:
            tokens = F.normalize(tokens, dim=-1)
            keys = F.normalize(keys, dim=-1)
        attn = F.softmax(self.log_beta.exp() * tokens @ keys.T, dim=-1)
        retrieved = F.fold(
            (attn @ self.memories).transpose(1, 2), (H, W), self.p, stride=self.stride
        )
        if self.stride != self.p:
            # average where patches overlap
            ones = torch.ones(1, 1, H, W, device=logits.device, dtype=logits.dtype)
            counts = F.fold(
                F.unfold(ones, self.p, stride=self.stride),
                (H, W),
                self.p,
                stride=self.stride,
            )
            retrieved = retrieved / counts
        return unary + self.gate * retrieved


class RecurrentSegmentation(nn.Module):
    """Feed-forward model (e.g. smp.Unet) + a recurrent block applied n_iter times on its logits."""

    def __init__(self, base_model, block, n_iter=5):
        super().__init__()
        self.base_model = base_model
        self.block = block
        self.n_iter = n_iter

    def forward(self, x, n_iter=None, return_all=False):
        # return_all: list with the logits after 0 (base model), 1, ..., n_iter steps
        n_iter = self.n_iter if n_iter is None else n_iter
        unary = self.base_model(x)
        logits = unary
        outputs = [unary]
        for _ in range(n_iter):
            logits = self.block(logits, unary, x)
            outputs.append(logits)
        return outputs if return_all else logits


class DecoderFeatureRecurrentSegmentation(nn.Module):
    """smp model + a recurrent block applied n_iter times on its logits; the block also
    sees the last decoder feature map (the input of the segmentation head)."""

    def __init__(self, base_model, block, n_iter=5):
        super().__init__()
        self.base_model = base_model
        self.block = block
        self.n_iter = n_iter

    def forward(self, x, n_iter=None, return_all=False):
        # return_all: list with the logits after 0 (base model), 1, ..., n_iter steps
        n_iter = self.n_iter if n_iter is None else n_iter
        features = self.base_model.decoder(self.base_model.encoder(x))
        unary = self.base_model.segmentation_head(features)
        logits = unary
        outputs = [unary]
        for _ in range(n_iter):
            logits = self.block(logits, unary, features)
            outputs.append(logits)
        return outputs if return_all else logits


def resize_to(t, size):
    # average-pool when shrinking, bilinear when enlarging
    if t.shape[-2] >= size[0]:
        return F.adaptive_avg_pool2d(t, size)
    return F.interpolate(t, size, mode="bilinear", align_corners=False)


class ConvGRUBlock(nn.Module):
    """One step of iterative refinement (RAFT-style): a convolutional GRU updates a hidden
    state from the decoder features (context) and the current probabilities; the logits
    are the base logits plus a correction read from the hidden state. Every step the hidden
    state sees ~kernel_size more pixels (on the downsampled grid), so information travels
    further as T grows, and the GRU gates keep the hidden state bounded."""

    def __init__(
        self, feature_channels, num_classes=2, hidden=32, kernel_size=5, downsample=2
    ):
        super().__init__()
        self.downsample = downsample  # 2 → GRU on a 256x256 grid for 512x512 inputs
        p = kernel_size // 2
        self.context = nn.Conv2d(feature_channels, hidden, 3, padding=1)
        self.init_hidden = nn.Conv2d(feature_channels, hidden, 3, padding=1)
        in_channels = hidden + num_classes  # context + current probabilities
        self.convz = nn.Conv2d(hidden + in_channels, hidden, kernel_size, padding=p)
        self.convr = nn.Conv2d(hidden + in_channels, hidden, kernel_size, padding=p)
        self.convq = nn.Conv2d(hidden + in_channels, hidden, kernel_size, padding=p)
        self.delta = nn.Conv2d(hidden, num_classes, 3, padding=1)
        nn.init.zeros_(self.delta.weight)  # starts identical to the base model
        nn.init.zeros_(self.delta.bias)

    def init_state(self, features, size):
        # context (fixed for all steps) and initial hidden state
        f = resize_to(
            features, (size[0] // self.downsample, size[1] // self.downsample)
        )
        return F.relu(self.context(f)), torch.tanh(self.init_hidden(f))

    def forward(self, logits, unary, hidden, context):
        L = logits.shape[1]
        probs = resize_to(F.softmax(logits, 1) - 1.0 / L, hidden.shape[-2:])
        inputs = torch.cat([context, probs], 1)
        hx = torch.cat([hidden, inputs], 1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * hidden, inputs], 1)))
        hidden = (1 - z) * hidden + z * q
        delta = F.interpolate(
            self.delta(hidden),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # Anchored to the base logits: the output stays bounded however large T is
        return unary + delta, hidden


class GRURefinementSegmentation(nn.Module):
    """smp model + ConvGRUBlock applied n_iter times. With random_n_iter, every training
    batch uses a random number of steps in 1..n_iter, so every step is trained to give a
    good output (and the model is not tied to one T).
    context_from_image: the block sees only the image (not the decoder features), so it has
    to propagate information step by step, like a CRF. supervise_unary=False: with deep
    supervision, no loss on step 0, so the base model gives a proposal that the block refines."""

    def __init__(
        self,
        base_model,
        block,
        n_iter=8,
        random_n_iter=True,
        context_from_image=False,
        supervise_unary=True,
    ):
        super().__init__()
        self.base_model = base_model
        self.block = block
        self.n_iter = n_iter
        self.random_n_iter = random_n_iter
        self.context_from_image = context_from_image
        self.supervise_unary = supervise_unary  # read by LitSegModel.training_step

    def forward(self, x, n_iter=None, return_all=False):
        # return_all: list with the logits after 0 (base model), 1, ..., n_iter steps
        if n_iter is None:
            n_iter = self.n_iter
            if self.training and self.random_n_iter:
                n_iter = int(torch.randint(1, self.n_iter + 1, (1,)))
        features = self.base_model.decoder(self.base_model.encoder(x))
        unary = self.base_model.segmentation_head(features)
        context, hidden = self.block.init_state(
            x if self.context_from_image else features, unary.shape[-2:]
        )
        logits = unary
        outputs = [unary]
        for _ in range(n_iter):
            logits, hidden = self.block(logits, unary, hidden, context)
            outputs.append(logits)
        return outputs if return_all else logits


# Default encoder for SegFormer (it only works with Mix Transformer encoders: mit_b0 ... mit_b5)
SEGFORMER_ENCODER = "mit_b2"


def segmentation_model_selector(
    arg_model,
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    num_classes=2,
    n_iter=5,
    block_kwargs=None,  # extra arguments for CRFBlock / HopfieldBlock (e.g. kernel_size)
):
    arg_model = arg_model.lower()
    block_kwargs = block_kwargs or {}

    # arg_model = "<base>" or "<base>_crf" / "<base>_hopfield" (block on the output logits),
    # "<base>_mf_hopfield" (Hopfield on the softmax probabilities, mean-field form),
    # "<base>_kv_hopfield" (keys from the decoder features, values = label patches),
    # "<base>_gru" (iterative refinement with a convolutional GRU),
    # "<base>_gru_local" (ConvGRU on the image only, no loss on the base model's output),
    # or "<base>_bottleneck_crf" / "<base>_bottleneck_hopfield" (block on the encoder bottleneck)
    base_name, _, block_name = arg_model.partition("_")

    if base_name == "unet":
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    elif base_name == "segformer":
        if not encoder_name.startswith("mit_"):
            encoder_name = SEGFORMER_ENCODER
        base = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    else:
        raise ValueError(f"unknown model: {arg_model}")

    # Returns base model
    if block_name == "":
        return base

    # Returns base + CRF
    elif block_name == "crf":
        return RecurrentSegmentation(
            base, CRFBlock(num_classes, **block_kwargs), n_iter=n_iter
        )

    # Returns base + Hopfield Block
    elif block_name == "hopfield":
        return RecurrentSegmentation(
            base, HopfieldBlock(num_classes, **block_kwargs), n_iter=n_iter
        )

    # Returns base + GRU
    elif block_name == "gru":
        # Iterative refinement with a convolutional GRU (RAFT-style)
        channels = base.segmentation_head[0].in_channels
        return GRURefinementSegmentation(
            base, ConvGRUBlock(channels, num_classes, **block_kwargs), n_iter=n_iter
        )

    # Returns base + GRU local
    elif block_name == "gru_local":
        # ConvGRU that sees only the image + probabilities (a learned CRF), no loss on step 0
        return GRURefinementSegmentation(
            base,
            ConvGRUBlock(in_channels, num_classes, **block_kwargs),
            n_iter=n_iter,
            context_from_image=True,
            supervise_unary=False,
        )

    raise ValueError(f"unknown model: {arg_model}")
