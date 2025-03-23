import numpy as np
import matplotlib.pyplot as plt
import librosa
import moviepy.editor as mpe
from moviepy.editor import VideoClip
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import cv2

def process_audio_batch(batch_data):
    """
    Process a batch of audio frames to calculate amplitudes for visualization.
    
    Parameters:
    -----------
    batch_data: tuple (start_frame, end_frame, samples_per_frame, y, sr, n_dots)
    
    Returns:
    --------
    List of amplitude arrays for each frame in the batch
    """
    start_frame, end_frame, samples_per_frame, y, sr, n_dots = batch_data
    batch_results = []
    
    chunk_size = samples_per_frame // n_dots
    
    for frame_idx in range(start_frame, end_frame):
        start_sample = frame_idx * samples_per_frame
        end_sample = min(start_sample + samples_per_frame, len(y))
        
        if start_sample >= len(y):
            # If we're past the end of the audio, return zeros
            batch_results.append(np.zeros(n_dots))
            continue
            
        segment = y[start_sample:end_sample]
        
        # Calculate amplitudes for each dot
        amplitudes = np.zeros(n_dots)
        for j in range(n_dots):
            start_idx = j * chunk_size
            end_idx = min((j+1) * chunk_size, len(segment))
            if start_idx < len(segment):
                chunk = segment[start_idx:end_idx]
                # Use absolute amplitude mean
                amplitudes[j] = np.abs(chunk).mean()
                
        batch_results.append(amplitudes)
        
    return batch_results

def smooth_amplitudes(amplitudes_list, window_size=3):
    """Apply temporal smoothing to amplitudes between frames using vectorized operations."""
    if window_size <= 1:
        return amplitudes_list
        
    # Convert to numpy array for faster operations
    amplitudes_array = np.array(amplitudes_list)
    smoothed = np.zeros_like(amplitudes_array)
    half_window = window_size // 2
    
    for i in range(len(amplitudes_array)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(amplitudes_array), i + half_window + 1)
        
        # Create weight array (more weight to current frame)
        weights = np.ones(end_idx - start_idx)
        center_idx = i - start_idx
        if center_idx >= 0 and center_idx < len(weights):
            weights = weights * 0.5 / (len(weights) - 1)
            weights[center_idx] = 0.5
        else:
            weights = weights / len(weights)
            
        # Apply weights
        weighted_sum = np.sum(amplitudes_array[start_idx:end_idx] * weights[:, np.newaxis], axis=0)
        smoothed[i] = weighted_sum
        
    return smoothed

def create_dot_visualization_video(input_file, output_file=None, color='#00FFFF', quality='medium', 
                                   num_workers=None, max_duration=None, dot_size=6, dot_spacing=6,
                                   max_height_percent=30):
    """
    Creates a minimalist dot visualization video from an audio file
    with optimized performance.
    
    Parameters:
    -----------
    input_file       : path to input audio file
    output_file      : path to output video file (optional)
    color            : dot color (hex code or name)
    quality          : 'low', 'medium', or 'high' (affects FPS and resolution)
    num_workers      : number of CPU cores to use (None = auto-detect)
    max_duration     : maximum duration to process in seconds (None = auto-cap at 10 min)
    dot_size         : size of dots in pixels
    dot_spacing      : spacing between dots in pixels
    max_height_percent: maximum height of visualization as a percentage of frame height
    """
    # Generate output filename if not provided
    if output_file is None:
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_dots.mp4"

    print(f"Processing audio file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Dot color: {color}")
    
    # Determine CPU cores to use
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} CPU cores for processing")
    
    # Load the audio file
    print("Loading audio file...")
    # Use a lower sample rate for longer files to save memory and processing time
    y, sr = librosa.load(input_file, sr=None, mono=True)
    
    # Get the duration of the audio
    audio_duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Apply duration cap if requested
    if max_duration is not None and audio_duration > max_duration:
        duration = max_duration
        print(f"Capping visualization to {duration:.2f} seconds as requested")
    # Default cap at 10 minutes
    elif audio_duration > 600:
        duration = 600
        print(f"Audio is very long ({audio_duration:.2f}s). Capping visualization to first 10 minutes.")
    else:
        duration = audio_duration
    
    # Quality settings
    if quality == 'low':
        fps = 10
        height = 480
        width = 854
        smoothing_window = 3
    elif quality == 'high':
        fps = 30
        height = 1080
        width = 1920
        smoothing_window = 5
    else:  # medium (default)
        fps = 20
        height = 720
        width = 1280
        smoothing_window = 5
    
    print(f"Quality: {quality} (FPS={fps}, Resolution={width}x{height})")
    
    total_frames = int(fps * duration)
    samples_per_frame = int(sr / fps)
    
    # Calculate number of dots based on width and spacing
    n_dots = width // (dot_size + dot_spacing)
    
    # Pre-compute all amplitudes using parallel processing
    print("Pre-computing audio data using parallel processing...")
    
    batch_size = max(10, total_frames // (num_workers * 2))  # Aim for ~2 batches per worker
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare batches for parallel processing
        batch_args = []
        for batch in range(total_batches):
            start_frame = batch * batch_size
            end_frame = min((batch + 1) * batch_size, total_frames)
            batch_args.append((
                start_frame, end_frame, samples_per_frame, y, sr, n_dots
            ))
        
        # Process batches in parallel
        results = list(executor.map(process_audio_batch, batch_args))
        
    # Flatten results
    all_amplitudes = []
    for batch_result in results:
        all_amplitudes.extend(batch_result)
    
    # Apply smoothing for a cleaner animation
    print("Applying temporal smoothing...")
    all_amplitudes = smooth_amplitudes(all_amplitudes, window_size=smoothing_window)
    
    # Convert to numpy array for vectorized operations
    all_amplitudes = np.array(all_amplitudes)
    
    # Normalize amplitudes globally
    max_amp = np.max(all_amplitudes)
    if max_amp > 0:
        all_amplitudes = all_amplitudes / max_amp
    
    print("Finished pre-computing audio data")
    
    # Pre-calculate constants
    radius = (dot_size + 1) // 2
    center_y = height // 2
    max_viz_height = int(height * max_height_percent / 100)
    
    # Pre-calculate x positions
    x_positions = np.arange(n_dots) * (dot_size + dot_spacing) + dot_size // 2
    
    # Convert hex color to RGB
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Initialize frame and overlay
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    overlay = np.zeros_like(frame)
    
    # Define helper function to draw symmetric dots
    def draw_symmetric_dots(frame, x_coords, y1_coords, y2_coords, radius, color_full, color_half):
        """Draw dots with random full/half opacity at symmetric positions"""
        h, w = frame.shape[:2]
        y1_coords = np.clip(y1_coords, radius, h-radius)
        y2_coords = np.clip(y2_coords, radius, h-radius)
        
        # Create masks for full and half opacity
        mask = np.random.random(len(x_coords)) < 0.5
        overlay.fill(0)  # Clear overlay
        
        # Draw dots with full opacity
        full_mask = mask
        if np.any(full_mask):
            for x, y1, y2 in zip(x_coords[full_mask], y1_coords[full_mask], y2_coords[full_mask]):
                cv2.circle(overlay, (int(x), int(y1)), radius, rgb_color, -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(x), int(y2)), radius, rgb_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)
        
        # Draw dots with half opacity
        overlay.fill(0)  # Clear overlay
        half_mask = ~mask
        if np.any(half_mask):
            for x, y1, y2 in zip(x_coords[half_mask], y1_coords[half_mask], y2_coords[half_mask]):
                cv2.circle(overlay, (int(x), int(y1)), radius, rgb_color, -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(x), int(y2)), radius, rgb_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)
    
    def make_frame(t):
        """Generate a frame for the given timestamp"""
        frame_idx = int(t * fps)
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
            
        # Clear frame
        frame.fill(0)
        
        if frame_idx < len(all_amplitudes):
            frame_amplitudes = all_amplitudes[frame_idx]
            
            # Calculate all y positions for symmetric dots
            y_offsets = np.minimum(frame_amplitudes * max_viz_height // 2, max_viz_height // 2)
            n_symmetric_dots = (y_offsets // (dot_size + dot_spacing)).astype(int) + 1
            
            # Draw center dots
            overlay.fill(0)
            mask = np.random.random(len(x_positions)) < 0.5
            
            # Draw full opacity center dots
            for x in x_positions[mask]:
                cv2.circle(overlay, (int(x), center_y), radius, rgb_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)
            
            # Draw half opacity center dots
            overlay.fill(0)
            for x in x_positions[~mask]:
                cv2.circle(overlay, (int(x), center_y), radius, rgb_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)
            
            # Draw symmetric dots for each level
            max_dots = int(n_symmetric_dots.max())
            for j in range(1, max_dots):
                y_pos = j * (dot_size + dot_spacing)
                valid_dots = j < n_symmetric_dots
                
                if np.any(valid_dots):
                    x_valid = x_positions[valid_dots]
                    draw_symmetric_dots(
                        frame, x_valid,
                        np.full_like(x_valid, center_y + y_pos),
                        np.full_like(x_valid, center_y - y_pos),
                        radius, rgb_color, rgb_color
                    )
        
        return frame.copy()
    
    print("Generating video...")
    # Create video clip using make_frame function
    animation_clip = VideoClip(make_frame, duration=duration)
    
    # Add audio
    audio_clip = mpe.AudioFileClip(input_file).subclip(0, duration)
    final_clip = animation_clip.set_audio(audio_clip)
    
    # Determine video bitrate based on quality
    if quality == 'low':
        bitrate = "2000k"
    elif quality == 'high':
        bitrate = "8000k"
    else:
        bitrate = "4000k"
    
    print(f"Writing video to {output_file}...")
    final_clip.write_videofile(
        output_file,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        bitrate=bitrate,
        threads=num_workers  # Use multiple threads for video encoding
    )
    
    print("Visualization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a minimalist dot visualization video from audio"
    )
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("-o", "--output_file", help="Path to output MP4 file (optional)")
    parser.add_argument("-c", "--color", default="#00FFFF",
                        help="Dot color (hex code or color name, default: #00FFFF)")
    parser.add_argument("-q", "--quality", default="medium", choices=["low", "medium", "high"],
                        help="Output quality (affects resolution and FPS)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of CPU cores to use (default: auto-detect)")
    parser.add_argument("-d", "--max_duration", type=float, default=None,
                        help="Maximum duration to process in seconds")
    parser.add_argument("--dot_size", type=int, default=6,
                        help="Size of dots in pixels (default: 6)")
    parser.add_argument("--dot_spacing", type=int, default=6,
                        help="Spacing between dots in pixels (default: 6)")
    parser.add_argument("--max_height", type=int, default=30,
                        help="Maximum height of visualization as percentage of frame height (default: 30)")
    
    args = parser.parse_args()
    create_dot_visualization_video(
        args.input_file, 
        args.output_file, 
        args.color,
        args.quality,
        args.workers,
        args.max_duration,
        args.dot_size,
        args.dot_spacing,
        args.max_height
    )
