from typing import Any, Optional, Sequence, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleicl.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_vae_text_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import VideoAllQFormer

logger = get_logger(__name__)


class SimpleICLActionOnly(nn.Module):
    """Action diffusion model conditioned on first-frame and video_all tokens."""

    is_action_only = True

    def __init__(
        self,
        action_expert: ActionDiT,
        qformer: VideoAllQFormer,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        share_vae_cond: bool = False,
    ):
        super().__init__()
        self.action_expert = action_expert
        self.dit = action_expert
        self.qformer = qformer
        self.vae = vae
        self.share_vae_cond = bool(share_vae_cond)
        self.vae_cond = self.vae if self.share_vae_cond else copy.deepcopy(vae)
        self.register_buffer(
            "_vae_cond_shared_marker",
            torch.tensor(1 if self.share_vae_cond else 0, dtype=torch.uint8),
            persistent=True,
        )
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
            if self.proprio_dim is not None
            else None
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.train_scheduler = self.train_action_scheduler
        self.infer_scheduler = self.infer_action_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_action = float(loss_lambda_action)
        self.model_paths: dict[str, Any] = {}
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        qformer_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        share_vae_cond: bool = False,
    ):
        if not isinstance(qformer_config, dict):
            raise ValueError("`qformer_config` must be provided as a dict.")
        if not isinstance(action_dit_config, dict):
            raise ValueError("`action_dit_config` must be provided as a dict.")
        assets = load_wan22_vae_text_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            load_text_encoder=load_text_encoder,
        )
        qformer = VideoAllQFormer(**qformer_config).to(device=device, dtype=torch_dtype)
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(
            action_dit_config.get("video_all_context_dim", qformer.hidden_dim)
        ) != int(qformer.hidden_dim):
            raise ValueError(
                "Action-only `action_dit_config.video_all_context_dim` must equal "
                f"qformer hidden_dim ({qformer.hidden_dim})."
            )
        if text_dim is None:
            if assets["text_encoder"] is None:
                raise ValueError(
                    "`text_dim` is required when `load_text_encoder=False`."
                )
            text_dim = int(assets["text_encoder"].dim)
        model = cls(
            action_expert=action_expert,
            qformer=qformer,
            vae=assets["vae"],
            text_encoder=assets["text_encoder"],
            tokenizer=assets["tokenizer"],
            text_dim=int(text_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_action=loss_lambda_action,
            share_vae_cond=share_vae_cond,
        )
        model.model_paths = {
            "vae": assets["vae_path"],
            "text_encoder": assets["text_encoder_path"],
            "tokenizer": assets["tokenizer_path"],
            "action_dit_backbone": "SKIPPED_PRETRAIN"
            if skip_dit_load_from_pretrain
            else action_dit_pretrained_path,
        }
        return model

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Prompt encoding requires loaded text encoder/tokenizer.")
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(self.device), torch.ones_like(mask)

    def _append_proprio_to_context(self, context, context_mask, proprio):
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` must be [B, {self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        token = self.proprio_encoder(
            proprio.to(self.device, dtype=context.dtype).unsqueeze(1)
        )
        mask = torch.ones(
            (context.shape[0], 1), dtype=torch.bool, device=context.device
        )
        return torch.cat([context, token], 1), torch.cat([context_mask, mask], 1)

    @staticmethod
    def _module_requires_grad(module):
        return any(p.requires_grad for p in module.parameters())

    def _unshare_vae_cond(self, reason: str):
        if self.vae_cond is self.vae:
            self.vae_cond = copy.deepcopy(self.vae)
            self.share_vae_cond = False
            self._vae_cond_shared_marker.fill_(0)
            logger.warning("Unsharing condition VAE %s.", reason)

    def _encode_cond_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self._module_requires_grad(self.vae_cond)
        ):
            return self.vae_cond.encode(
                video_tensor,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

    def _encode_first_frame_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        first = video_tensor[:, :, :1].to(self.device, dtype=self.torch_dtype)
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self._module_requires_grad(self.vae)
        ):
            latents = self.vae.encode(
                first,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if isinstance(latents, (list, tuple)):
            latents = torch.stack(latents)
        return latents

    def _first_frame_tokens(self, latents):
        return self.qformer._patchify(self.qformer.video_all_embedding, latents)

    def _build_context(
        self,
        context,
        context_mask,
        first_frame_latents,
        video_all_latents,
        proprio=None,
    ):
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError("`context/context_mask` must be [B,L,D]/[B,L].")
        context = context.to(self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(self.device, dtype=torch.bool)
        context, context_mask = self._append_proprio_to_context(
            context, context_mask, proprio
        )
        if first_frame_latents is None or video_all_latents is None:
            raise ValueError(
                "Action-only model requires first-frame and video_all latents."
            )
        first_tokens = self._first_frame_tokens(first_frame_latents)
        condition_tokens = self.qformer.encode_query(video_all_latents)
        first_mask = torch.ones(
            first_tokens.shape[:2], dtype=torch.bool, device=first_tokens.device
        )
        condition_mask = torch.ones(
            condition_tokens.shape[:2], dtype=torch.bool, device=condition_tokens.device
        )
        return (
            context,
            context_mask,
            first_tokens,
            first_mask,
            condition_tokens,
            condition_mask,
        )

    def build_inputs(self, sample, tiled=False):
        video = sample["video"]
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}"
            )
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for action-only training.")
        context, context_mask = sample["context"], sample["context_mask"]
        proprio = sample.get("proprio")
        if proprio is not None:
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be [B,T,D], got {tuple(proprio.shape)}"
                )
            proprio = proprio[:, 0, :]
        video_all = sample.get("video_all")
        if video_all is None:
            raise ValueError(
                "Action-only training requires `sample['video_all']` for Q-Former conditioning."
            )
        video = video.to(self.device, dtype=self.torch_dtype)
        video_all = video_all.to(self.device, dtype=self.torch_dtype)
        first_latents = self._encode_first_frame_latents(video, tiled=tiled)
        cond_latents = self._encode_cond_latents(video_all, tiled=tiled)
        (
            context,
            context_mask,
            first_tokens,
            first_mask,
            condition_tokens,
            condition_mask,
        ) = self._build_context(
            context, context_mask, first_latents, cond_latents, proprio
        )
        action = sample["action"].to(self.device, dtype=self.torch_dtype)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)
        return {
            "context": context,
            "context_mask": context_mask,
            "first_frame_context": first_tokens,
            "first_frame_context_mask": first_mask,
            "video_all_context": condition_tokens,
            "video_all_context_mask": condition_mask,
            "action": action,
            "action_is_pad": action_is_pad,
        }

    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        action = inputs["action"]
        batch_size = action.shape[0]
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        pred = self.action_expert(
            action_tokens=noisy,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            first_frame_context=inputs["first_frame_context"],
            first_frame_context_mask=inputs["first_frame_context_mask"],
            video_all_context=inputs["video_all_context"],
            video_all_context_mask=inputs["video_all_context_mask"],
        )
        token_loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(
            dim=2
        )
        pad = inputs["action_is_pad"]
        if pad is not None:
            valid = (~pad).to(token_loss.dtype)
            token_loss = (token_loss * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        else:
            token_loss = token_loss.mean(1)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            token_loss.dtype
        )
        loss = self.loss_lambda_action * (token_loss * weight).mean()
        return loss, {"loss_action": float(loss.detach().item())}

    @torch.no_grad()
    def infer_action(
        self,
        prompt,
        input_image,
        action_horizon,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale=1.0,
        num_inference_steps=20,
        sigma_shift=None,
        seed=None,
        rand_device="cpu",
        tiled=False,
        video_all_latents=None,
    ):
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError("`input_image` must be [1,3,H,W] or [3,H,W].")
        if context is None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context.to(self.device, dtype=self.torch_dtype),
                context_mask.to(self.device, dtype=torch.bool),
                proprio.to(self.device, dtype=self.torch_dtype),
            )
        first = input_image.to(self.device, dtype=self.torch_dtype).unsqueeze(2)
        first_latents = self.vae.encode(first, device=self.device, tiled=tiled)
        if isinstance(first_latents, (list, tuple)):
            first_latents = torch.stack(first_latents)
        if video_all_latents is None:
            raise ValueError(
                "`video_all_latents` is required for Action-only inference."
            )
        _, _, first_tokens, first_mask, condition_tokens, condition_mask = (
            self._build_context(
                context, context_mask, first_latents, video_all_latents, proprio=None
            )
        )
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, dtype=self.torch_dtype)
        ts, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for t, delta in zip(ts, deltas):
            timestep = t.unsqueeze(0).to(self.device, dtype=latents.dtype)
            pred = self.action_expert(
                action_tokens=latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                first_frame_context=first_tokens,
                first_frame_context_mask=first_mask,
                video_all_context=condition_tokens,
                video_all_context_mask=condition_mask,
            )
            latents = self.infer_action_scheduler.step(pred, delta, latents)
        return {"action": latents[0].detach().float().cpu()}

    def infer(
        self,
        input_image,
        action_horizon,
        num_frames=None,
        video_all_latents=None,
        **kwargs,
    ):
        return self.infer_action(
            input_image=input_image,
            action_horizon=action_horizon,
            video_all_latents=video_all_latents,
            **kwargs,
        )

    def load_vae_cond_pretrained(self, path):
        payload = torch.load(path, map_location=self.device)
        state_dict = payload.get("vae_cond") if isinstance(payload, dict) else None
        if (
            state_dict is None
            and isinstance(payload, dict)
            and all(isinstance(v, torch.Tensor) for v in payload.values())
        ):
            state_dict = payload
        if state_dict is None:
            raise ValueError(f"Checkpoint {path} has no `vae_cond` weights.")
        self._unshare_vae_cond("before loading independent checkpoint weights")
        self.vae_cond.load_state_dict(state_dict, strict=False)
        return payload

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "action_expert": self.action_expert.state_dict(),
            "qformer": self.qformer.state_dict(),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self._module_requires_grad(self.vae_cond):
            payload["vae_cond"] = self.vae_cond.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = torch.load(path, map_location=self.device)
        if "action_expert" not in payload or "qformer" not in payload:
            raise ValueError(f"Checkpoint missing action-only weights: {path}")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.qformer.load_state_dict(payload["qformer"], strict=True)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        if "vae_cond" in payload:
            self._unshare_vae_cond("before loading checkpoint condition weights")
            self.vae_cond.load_state_dict(payload["vae_cond"], strict=False)
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
