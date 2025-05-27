#!/usr/bin/env python3

"""
Split videos into scenes using PySceneDetect.

This script provides a command-line interface for splitting videos into scenes using various detection algorithms.
It supports multiple detection methods, preview image generation, and customizable parameters for fine-tuning
the scene detection process.

Basic usage:
    # Split all videos in a directory using default content-based detection
    scenes_split.py input_dir/ output_dir/

    # Save 3 preview images per scene
    scenes_split.py input_dir/ output_dir/ --save-images 3

    # Process specific duration and filter short scenes
    scenes_split.py input_dir/ output_dir/ --duration 60s --filter-shorter-than 2s

Advanced usage:
    # Content detection with minimum scene length and frame skip
    scenes_split.py input_dir/ output_dir/ --detector content --min-scene-length 30 --frame-skip 2

    # Use adaptive detection with custom detector and detector parameters
    scenes_split.py input_dir/ output_dir/ --detector adaptive --threshold 3.0 --adaptive-window 10
"""

import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    HistogramDetector,
    SceneManager,
    ThresholdDetector,
    open_video,
)
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneDetector, write_scene_list_html
from scenedetect.scene_manager import save_images as save_scene_images
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg

app = typer.Typer(help="Split videos into scenes using PySceneDetect.")

# Common video file extensions
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
}


class DetectorType(str, Enum):
    """Available scene detection algorithms."""

    CONTENT = "content"  # Detects fast cuts using HSV color space
    ADAPTIVE = "adaptive"  # Detects fast two-phase cuts
    THRESHOLD = "threshold"  # Detects fast cuts/slow fades in from and out to a given threshold level
    HISTOGRAM = (
        "histogram"  # Detects based on YUV histogram differences in adjacent frames
    )


def create_detector(
    detector_type: DetectorType,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    luma_only: Optional[bool] = None,
    adaptive_window: Optional[int] = None,
    fade_bias: Optional[float] = None,
) -> SceneDetector:
    """Create a scene detector based on the specified type and parameters.

    Args:
        detector_type: Type of detector to create
        threshold: Detection threshold (meaning varies by detector)
        min_scene_len: Minimum scene length in frames
        luma_only: If True, only use brightness for content detection
        adaptive_window: Window size for adaptive detection
        fade_bias: Bias for fade in/out detection (-1.0 to 1.0)

    Note: Parameters set to None will use the detector's built-in default values.

    Returns:
        Configured scene detector instance
    """
    # Set common arguments
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold

    if min_scene_len is not None:
        kwargs["min_scene_len"] = min_scene_len

    match detector_type:
        case DetectorType.CONTENT:
            if luma_only is not None:
                kwargs["luma_only"] = luma_only
            return ContentDetector(**kwargs)
        case DetectorType.ADAPTIVE:
            if adaptive_window is not None:
                kwargs["window_width"] = adaptive_window
            if luma_only is not None:
                kwargs["luma_only"] = luma_only
            if "threshold" in kwargs:
                # Special case for adaptive detector which uses different param name
                kwargs["adaptive_threshold"] = kwargs.pop("threshold")
            return AdaptiveDetector(**kwargs)
        case DetectorType.THRESHOLD:
            if fade_bias is not None:
                kwargs["fade_bias"] = fade_bias
            return ThresholdDetector(**kwargs)
        case DetectorType.HISTOGRAM:
            return HistogramDetector(**kwargs)
        case _:
            raise ValueError(f"Unknown detector type: {detector_type}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate and create output directory if it doesn't exist.

    Args:
        output_dir: Path to the output directory

    Returns:
        Path object of the validated output directory
    """
    path = Path(output_dir)

    if path.exists() and not path.is_dir():
        raise typer.BadParameter(f"{output_dir} exists but is not a directory")

    return path


def parse_timecode(video: any, time_str: Optional[str]) -> Optional[FrameTimecode]:
    """Parse a timecode string into a FrameTimecode object.

    Supports formats:
    - Frames: '123'
    - Seconds: '123s' or '123.45s'
    - Timecode: '00:02:03' or '00:02:03.456'

    Args:
        video: Video object to get framerate from
        time_str: String to parse, or None

    Returns:
        FrameTimecode object or None if input is None
    """
    if time_str is None:
        return None

    try:
        if time_str.endswith("s"):
            # Seconds format
            seconds = float(time_str[:-1])
            return FrameTimecode(timecode=seconds, fps=video.frame_rate)
        elif ":" in time_str:
            # Timecode format
            return FrameTimecode(timecode=time_str, fps=video.frame_rate)
        else:
            # Frame number format
            return FrameTimecode(timecode=int(time_str), fps=video.frame_rate)
    except ValueError as e:
        raise typer.BadParameter(
            f"Invalid timecode format: {time_str}. Use frames (123), "
            f"seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn])",
        ) from e


def get_video_files(input_dir: Path) -> List[Path]:
    """Get all video files from the input directory.

    Args:
        input_dir: Path to the input directory

    Returns:
        List of video file paths
    """
    video_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(file_path)

    return sorted(video_files)  # Sort for consistent processing order


def create_video_output_dir(output_dir: Path, video_name: str) -> Path:
    """Create a subdirectory for a specific video's output.

    Args:
        output_dir: Base output directory
        video_name: Name of the video (without extension)

    Returns:
        Path to the video-specific output directory
    """
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    return video_output_dir


def detect_and_split_scenes(  # noqa: PLR0913
    video_path: str,
    output_dir: Path,
    detector_type: DetectorType,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    max_scenes: Optional[int] = None,
    filter_shorter_than: Optional[str] = None,
    skip_start: Optional[int] = None,  # noqa: ARG001
    skip_end: Optional[int] = None,  # noqa: ARG001
    save_images_per_scene: int = 0,
    stats_file: Optional[str] = None,
    luma_only: bool = False,
    adaptive_window: Optional[int] = None,
    fade_bias: Optional[float] = None,
    downscale_factor: Optional[int] = None,
    frame_skip: int = 0,
    duration: Optional[str] = None,
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Detect and split scenes in a video using the specified parameters.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save output split scenes.
        detector_type: Type of scene detector to use.
        threshold: Detection threshold.
        min_scene_len: Minimum scene length in frames.
        max_scenes: Maximum number of scenes to detect.
        filter_shorter_than: Filter out scenes shorter than this duration (frames/seconds/timecode)
        skip_start: Number of frames to skip at start.
        skip_end: Number of frames to skip at end.
        save_images_per_scene: Number of images to save per scene (0 to disable).
        stats_file: Path to save detection statistics (optional).
        luma_only: Only use brightness for content detection.
        adaptive_window: Window size for adaptive detection.
        fade_bias: Bias for fade detection (-1.0 to 1.0).
        downscale_factor: Factor to downscale frames by during detection.
        frame_skip: Number of frames to skip (i.e. process every 1 in N+1 frames,
            where N is frame_skip, processing only 1/N+1 percent of the video,
            speeding up the detection time at the expense of accuracy).
            frame_skip must be 0 (the default) when using a StatsManager.
        duration: How much of the video to process from start position.
            Can be specified as frames (123), seconds (123s/123.45s),
            or timecode (HH:MM:SS[.nnn]).

    Returns:
        List of detected scenes as (start, end) FrameTimecode pairs.
    """
    # Create video stream
    video = open_video(video_path, backend="opencv")

    # Parse duration if specified
    duration_tc = parse_timecode(video, duration)

    # Parse filter_shorter_than if specified
    filter_shorter_than_tc = parse_timecode(video, filter_shorter_than)

    # Initialize scene manager with optional stats manager
    stats_manager = StatsManager() if stats_file else None
    scene_manager = SceneManager(stats_manager)

    # Configure scene manager
    if downscale_factor:
        scene_manager.auto_downscale = False
        scene_manager.downscale = downscale_factor

    # Create and add detector
    detector = create_detector(
        detector_type=detector_type,
        threshold=threshold,
        min_scene_len=min_scene_len,
        luma_only=luma_only,
        adaptive_window=adaptive_window,
        fade_bias=fade_bias,
    )
    scene_manager.add_detector(detector)

    # Detect scenes
    typer.echo("Detecting scenes...")
    scene_manager.detect_scenes(
        video=video,
        show_progress=True,
        frame_skip=frame_skip,
        duration=duration_tc,
    )

    # Get scene list
    scenes = scene_manager.get_scene_list()

    # Filter out scenes that are too short if filter_shorter_than is specified
    if filter_shorter_than_tc:
        original_count = len(scenes)
        scenes = [
            (start, end)
            for start, end in scenes
            if (end.get_frames() - start.get_frames())
            >= filter_shorter_than_tc.get_frames()
        ]
        if len(scenes) < original_count:
            typer.echo(
                f"Filtered out {original_count - len(scenes)} scenes shorter "
                f"than {filter_shorter_than_tc.get_seconds():.1f} seconds "
                f"({filter_shorter_than_tc.get_frames()} frames)",
            )

    # Apply max scenes limit if specified
    if max_scenes and len(scenes) > max_scenes:
        typer.echo(
            f"Dropping last {len(scenes) - max_scenes} scenes to meet max_scenes ({max_scenes}) limit"
        )
        scenes = scenes[:max_scenes]

    # Print scene information
    typer.echo(f"Found {len(scenes)} scenes:")
    for i, (start, end) in enumerate(scenes, 1):
        typer.echo(
            f"Scene {i}: {start.get_timecode()} to {end.get_timecode()} "
            f"({end.get_frames() - start.get_frames()} frames)",
        )

    # Save stats if requested
    if stats_file:
        typer.echo(f"Saving detection stats to {stats_file}")
        stats_manager.save_to_csv(stats_file)

    # Split video into scenes
    typer.echo("Splitting video into scenes...")
    split_video_ffmpeg(
        input_video_path=video_path,
        scene_list=scenes,
        output_dir=output_dir,
        show_progress=True,
    )
    typer.echo(f"Scenes have been saved to: {output_dir}")

    # Save preview images if requested
    if save_images_per_scene > 0:
        typer.echo(f"Saving {save_images_per_scene} preview images per scene...")
        image_filenames = save_scene_images(
            scene_list=scenes,
            video=video,
            num_images=save_images_per_scene,
            output_dir=str(output_dir),
            show_progress=True,
        )

        # Generate HTML report with scene information and previews
        html_path = output_dir / "scene_report.html"
        write_scene_list_html(
            output_html_filename=str(html_path),
            scene_list=scenes,
            image_filenames=image_filenames,
        )
        typer.echo(f"Scene report saved to: {html_path}")

    return scenes


def process_single_video(  # noqa: PLR0913
    video_path: Path,
    output_dir: Path,
    detector_type: DetectorType,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    max_scenes: Optional[int] = None,
    filter_shorter_than: Optional[str] = None,
    skip_start: Optional[int] = None,
    skip_end: Optional[int] = None,
    save_images_per_scene: int = 0,
    stats_file_prefix: Optional[str] = None,
    luma_only: bool = False,
    adaptive_window: Optional[int] = None,
    fade_bias: Optional[float] = None,
    downscale_factor: Optional[int] = None,
    frame_skip: int = 0,
    duration: Optional[str] = None,
) -> bool:
    """Process a single video file with error handling.

    Args:
        video_path: Path to the video file
        output_dir: Base output directory
        stats_file_prefix: Prefix for stats file (video name will be appended)
        ... (other args same as detect_and_split_scenes)

    Returns:
        True if processing succeeded, False if it failed
    """
    video_name = video_path.stem
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Processing video: {video_path.name}")
    typer.echo(f"{'='*60}")

    try:
        # Create video-specific output directory
        video_output_dir = create_video_output_dir(output_dir, video_name)

        # Create stats file path if requested
        stats_file = None
        if stats_file_prefix:
            stats_file = f"{stats_file_prefix}_{video_name}.csv"

        # Process the video
        scenes = detect_and_split_scenes(
            video_path=str(video_path),
            output_dir=video_output_dir,
            detector_type=detector_type,
            threshold=threshold,
            min_scene_len=min_scene_len,
            max_scenes=max_scenes,
            filter_shorter_than=filter_shorter_than,
            skip_start=skip_start,
            skip_end=skip_end,
            duration=duration,
            save_images_per_scene=save_images_per_scene,
            stats_file=stats_file,
            luma_only=luma_only,
            adaptive_window=adaptive_window,
            fade_bias=fade_bias,
            downscale_factor=downscale_factor,
            frame_skip=frame_skip,
        )

        typer.echo(
            f"✅ Successfully processed {video_path.name} - {len(scenes)} scenes created"
        )
        return True

    except Exception as e:
        typer.echo(f"❌ Error processing {video_path.name}: {str(e)}", err=True)
        typer.echo(f"Continuing with next video...", err=True)
        return False


@app.command()
def main(  # noqa: PLR0913
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the input video file or directory containing video files",
        exists=True,
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directory where split scenes will be saved",
    ),
    detector: DetectorType = typer.Option(  # noqa: B008
        DetectorType.CONTENT,
        help="Scene detection algorithm to use",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        help="Detection threshold (meaning varies by detector)",
    ),
    max_scenes: Optional[int] = typer.Option(
        None,
        help="Maximum number of scenes to produce per video",
    ),
    min_scene_length: Optional[int] = typer.Option(
        None,
        help="Minimum scene length during detection. Forces the detector to make scenes at least this many frames. "
        "This affects scene detection behavior but does not filter out short scenes.",
    ),
    filter_shorter_than: Optional[str] = typer.Option(
        None,
        help="Filter out scenes shorter than this duration. Can be specified as frames (123), "
        "seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn]). These scenes will be detected but not saved.",
    ),
    skip_start: Optional[int] = typer.Option(
        None,
        help="Number of frames to skip at the start of each video",
    ),
    skip_end: Optional[int] = typer.Option(
        None,
        help="Number of frames to skip at the end of each video",
    ),
    duration: Optional[str] = typer.Option(
        None,
        "-d",
        help="How much of each video to process. Can be specified as frames (123), "
        "seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn])",
    ),
    save_images: int = typer.Option(
        0,
        help="Number of preview images to save per scene (0 to disable)",
    ),
    stats_file_prefix: Optional[str] = typer.Option(
        None,
        help="Prefix for detection statistics CSV files (video name will be appended)",
    ),
    luma_only: bool = typer.Option(
        False,
        help="Only use brightness for content detection",
    ),
    adaptive_window: Optional[int] = typer.Option(
        None,
        help="Window size for adaptive detection",
    ),
    fade_bias: Optional[float] = typer.Option(
        None,
        help="Bias for fade detection (-1.0 to 1.0)",
    ),
    downscale: Optional[int] = typer.Option(
        None,
        help="Factor to downscale frames by during detection",
    ),
    frame_skip: int = typer.Option(
        0,
        help="Number of frames to skip during processing",
    ),
) -> None:
    """Split videos into scenes using PySceneDetect.

    Can process either a single video file or all video files in a directory.
    When processing a directory, each video's output will be saved in a subdirectory
    named after the video file.
    """
    if skip_start or skip_end:
        typer.echo("Skipping start and end frames is not supported yet.")
        return

    # Validate output directory
    output_path = validate_output_dir(output_dir)

    # Determine if input is a file or directory
    if input_path.is_file():
        # Single file mode (backward compatibility)
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            typer.echo(
                f"Error: {input_path} does not appear to be a video file", err=True
            )
            raise typer.Exit(1)

        video_files = [input_path]
        typer.echo(f"Processing single video file: {input_path}")
    elif input_path.is_dir():
        # Directory mode
        video_files = get_video_files(input_path)
        if not video_files:
            typer.echo(f"No video files found in directory: {input_path}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Found {len(video_files)} video files in {input_path}")
        for video_file in video_files:
            typer.echo(f"  - {video_file.name}")
    else:
        typer.echo(f"Error: {input_path} is neither a file nor a directory", err=True)
        raise typer.Exit(1)

    # Process all video files
    successful_count = 0
    failed_count = 0

    for video_file in video_files:
        success = process_single_video(
            video_path=video_file,
            output_dir=output_path,
            detector_type=detector,
            threshold=threshold,
            min_scene_len=min_scene_length,
            max_scenes=max_scenes,
            filter_shorter_than=filter_shorter_than,
            skip_start=skip_start,
            skip_end=skip_end,
            duration=duration,
            save_images_per_scene=save_images,
            stats_file_prefix=stats_file_prefix,
            luma_only=luma_only,
            adaptive_window=adaptive_window,
            fade_bias=fade_bias,
            downscale_factor=downscale,
            frame_skip=frame_skip,
        )

        if success:
            successful_count += 1
        else:
            failed_count += 1

    # Print summary
    typer.echo(f"\n{'='*60}")
    typer.echo("PROCESSING SUMMARY")
    typer.echo(f"{'='*60}")
    typer.echo(f"Total videos processed: {len(video_files)}")
    typer.echo(f"Successful: {successful_count}")
    typer.echo(f"Failed: {failed_count}")

    if failed_count > 0:
        typer.echo(
            f"\n⚠️  {failed_count} video(s) failed to process. Check the error messages above."
        )
        raise typer.Exit(1)
    else:
        typer.echo(f"\n🎉 All videos processed successfully!")


if __name__ == "__main__":
    app()
