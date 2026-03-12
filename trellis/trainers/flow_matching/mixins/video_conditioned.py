import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from ....utils import dist_utils


class VideoConditionedMixin:
    """
    Encode a temporal stack of conditioning frames frame-by-frame with DINOv2.

    Expected conditioning tensors:
    - training: [B, T, C, H, W]
    - inference after flattening: [B*T, C, H, W]
    """

    def __init__(self, *args, image_cond_model: str = "dinov2_vitl14_reg", **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None

    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        if hasattr(super(VideoConditionedMixin, VideoConditionedMixin), "prepare_for_training"):
            super(VideoConditionedMixin, VideoConditionedMixin).prepare_for_training(**kwargs)
        torch.hub.load("facebookresearch/dinov2", image_cond_model, pretrained=True)

    def _init_image_cond_model(self):
        with dist_utils.local_master_first():
            if os.getenv("EXP_PLATFORM", "other") == "huoshanyun":
                dinov2_model = torch.hub.load(
                    "/fs-computility/mllm/lizhibing/.cache/torch/hub/facebookresearch_dinov2_main",
                    self.image_cond_model_name,
                    source="local",
                    pretrained=True,
                )
            else:
                dinov2_model = torch.hub.load("facebookresearch/dinov2", self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_cond_model = {
            "model": dinov2_model,
            "transform": transform,
        }

    @torch.no_grad()
    def encode_video(self, image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 5:
                batch_size, num_frames = image.shape[:2]
                image = image.reshape(batch_size * num_frames, *image.shape[2:])
            elif image.ndim != 4:
                raise ValueError(f"Unsupported conditioning tensor shape: {image.shape}")
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model["transform"](image).cuda()
        features = self.image_cond_model["model"](image, is_training=True)["x_prenorm"]
        return F.layer_norm(features, features.shape[-1:])

    def get_cond(self, cond, **kwargs):
        cond = self.encode_video(cond)
        kwargs["neg_cond"] = torch.zeros_like(cond)
        return super().get_cond(cond, **kwargs)

    def get_inference_cond(self, cond, **kwargs):
        cond = self.encode_video(cond)
        kwargs["neg_cond"] = torch.zeros_like(cond)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        if isinstance(cond, torch.Tensor) and cond.ndim == 5:
            return {"image": {"value": cond[:, 0], "type": "image"}}
        if isinstance(cond, torch.Tensor) and cond.ndim == 4:
            return {"image": {"value": cond, "type": "image"}}
        return {}
