import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
from diffusers.utils import BaseOutput, is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

import torch
from diffusers import (
    LTXConditionPipeline,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput


@dataclass
class LTXPipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


@dataclass
class LTXVideoCondition:
    """
    Defines a single frame-conditioning item for LTX Video - a single frame or a sequence of frames.

    Attributes:
        image (`PIL.Image.Image`):
            The image to condition the video on.
        video (`List[PIL.Image.Image]`):
            The video to condition the video on.
        frame_index (`int`):
            The frame index at which the image or video will conditionally effect the video generation.
        strength (`float`, defaults to `1.0`):
            The strength of the conditioning effect. A value of `1.0` means the conditioning effect is fully applied.
    """

    image: Optional[PIL.Image.Image] = None
    video: Optional[List[PIL.Image.Image]] = None
    frame_index: int = 0
    strength: float = 1.0


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# from LTX-Video/ltx_video/schedulers/rf.py
def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


class PatchedConditionPipeline(LTXConditionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        conditions: Union[LTXVideoCondition, List[LTXVideoCondition]] = None,
        image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        video: Optional[List[PipelineImageInput]] = None,
        frame_index: Union[int, List[int]] = 0,
        strength: Union[float, List[float]] = 1.0,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 704,
        num_frames: int = 161,
        frame_rate: int = 25,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 3,
        image_cond_noise_scale: float = 0.15,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            conditions (`List[LTXVideoCondition], *optional*`):
                The list of frame-conditioning items for the video generation.If not provided, conditions will be
                created using `image`, `video`, `frame_index` and `strength`.
            image (`PipelineImageInput` or `List[PipelineImageInput]`, *optional*):
                The image or images to condition the video generation. If not provided, one has to pass `video` or
                `conditions`.
            video (`List[PipelineImageInput]`, *optional*):
                The video to condition the video generation. If not provided, one has to pass `image` or `conditions`.
            frame_index (`int` or `List[int]`, *optional*):
                The frame index or frame indices at which the image or video will conditionally effect the video
                generation. If not provided, one has to pass `conditions`.
            strength (`float` or `List[float]`, *optional*):
                The strength or strengths of the conditioning effect. If not provided, one has to pass `conditions`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if latents is not None:
            raise ValueError("Passing latents is not yet supported.")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            conditions=conditions,
            image=image,
            video=video,
            frame_index=frame_index,
            strength=strength,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions]

            strength = [condition.strength for condition in conditions]
            frame_index = [condition.frame_index for condition in conditions]
            image = [condition.image for condition in conditions]
            video = [condition.video for condition in conditions]
        elif image is not None or video is not None:
            if not isinstance(image, list):
                image = [image]
                num_conditions = 1
            elif isinstance(image, list):
                num_conditions = len(image)
            if not isinstance(video, list):
                video = [video]
                num_conditions = 1
            elif isinstance(video, list):
                num_conditions = len(video)

            if not isinstance(frame_index, list):
                frame_index = [frame_index] * num_conditions
            if not isinstance(strength, list):
                strength = [strength] * num_conditions

        device = self._execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        vae_dtype = self.vae.dtype

        conditioning_tensors = []
        is_conditioning_image_or_video = image is not None or video is not None
        if is_conditioning_image_or_video:
            for (
                condition_image,
                condition_video,
                condition_frame_index,
                condition_strength,
            ) in zip(image, video, frame_index, strength):
                if condition_image is not None:
                    condition_tensor = (
                        self.video_processor.preprocess(condition_image, height, width)
                        .unsqueeze(2)
                        .to(device, dtype=vae_dtype)
                    )
                elif condition_video is not None:
                    condition_tensor = self.video_processor.preprocess_video(
                        condition_video, height, width
                    )
                    num_frames_input = condition_tensor.size(2)
                    num_frames_output = self.trim_conditioning_sequence(
                        condition_frame_index, num_frames_input, num_frames
                    )
                    condition_tensor = condition_tensor[:, :, :num_frames_output]
                    condition_tensor = condition_tensor.to(device, dtype=vae_dtype)
                else:
                    raise ValueError(
                        "Either `image` or `video` must be provided for conditioning."
                    )

                if condition_tensor.size(2) % self.vae_temporal_compression_ratio != 1:
                    raise ValueError(
                        f"Number of frames in the video must be of the form (k * {self.vae_temporal_compression_ratio} + 1) "
                        f"but got {condition_tensor.size(2)} frames."
                    )
                conditioning_tensors.append(condition_tensor)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, video_coords, extra_conditioning_num_latents = (
            self.prepare_latents(
                conditioning_tensors,
                strength,
                frame_index,
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        )

        video_coords = video_coords.float()
        video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)

        init_latents = latents.clone() if is_conditioning_image_or_video else None

        if self.do_classifier_free_guidance:
            video_coords = torch.cat([video_coords, video_coords], dim=0)

        # 5. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if image_cond_noise_scale > 0 and init_latents is not None:
                    # Add timestep-dependent noise to the hard-conditioning latents
                    # This helps with motion continuity, especially when conditioned on a single frame
                    latents = self.add_noise_to_image_conditioning_latents(
                        t / 1000.0,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                if is_conditioning_image_or_video:
                    conditioning_mask_model_input = (
                        torch.cat([conditioning_mask, conditioning_mask])
                        if self.do_classifier_free_guidance
                        else conditioning_mask
                    )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).unsqueeze(-1).float()
                if is_conditioning_image_or_video:
                    timestep = torch.min(
                        timestep, (1 - conditioning_mask_model_input) * 1000.0
                    )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    video_coords=video_coords,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    timestep, _ = timestep.chunk(2)

                denoised_latents = self.scheduler.step(
                    -noise_pred,
                    t,
                    latents,
                    per_token_timesteps=timestep,
                    return_dict=False,
                )[0]
                if is_conditioning_image_or_video:
                    tokens_to_denoise_mask = (
                        t / 1000 - 1e-6 < (1.0 - conditioning_mask)
                    ).unsqueeze(-1)
                    latents = torch.where(
                        tokens_to_denoise_mask, denoised_latents, latents
                    )
                else:
                    latents = denoised_latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if is_conditioning_image_or_video:
            latents = latents[:, extra_conditioning_num_latents:]

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        if output_type == "latent":
            video = latents
        else:
            latents = self._denormalize_latents(
                latents,
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = torch.randn(
                    latents.shape,
                    generator=generator,
                    device=device,
                    dtype=latents.dtype,
                )
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(
                    decode_timestep, device=device, dtype=latents.dtype
                )
                decode_noise_scale = torch.tensor(
                    decode_noise_scale, device=device, dtype=latents.dtype
                )[:, None, None, None, None]
                latents = (
                    1 - decode_noise_scale
                ) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
