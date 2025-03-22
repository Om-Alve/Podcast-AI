import numpy as np
import matplotlib.pyplot as plt
import librosa
import moviepy.editor as mpe
from moviepy.video.io.bindings import mplfig_to_npimage
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def compute_frame_bars_batch(batch_data):
    """
    Compute the height of each bar for a batch of animation frames.
    
    Parameters:
    -----------
    batch_data: tuple (start_frame, end_frame, samples_per_frame, y, sr, num_bars, weighting_factor)
    
    Returns:
    --------
    List of bar heights for each frame in the batch
    """
    start_frame, end_frame, samples_per_frame, y, sr, num_bars, weighting_factor = batch_data
    batch_results = []
    
    # Pre-calculate samples per bar
    total_samples = min(samples_per_frame, len(y) - start_frame * samples_per_frame)
    samples_per_bar = max(1, total_samples // num_bars)
    
    for frame_idx in range(start_frame, end_frame):
        start_sample = frame_idx * samples_per_frame
        end_sample = min(start_sample + samples_per_frame, len(y))
        
        if start_sample >= len(y):
            # If we're past the end of the audio, return zeros
            batch_results.append(np.zeros(num_bars))
            continue
            
        segment = y[start_sample:end_sample]
        
        # Vectorized bar height computation
        bar_heights = np.zeros(num_bars)
        for j in range(num_bars):
            start_idx = j * samples_per_bar
            end_idx = min((j+1) * samples_per_bar, len(segment))
            if start_idx < len(segment):
                chunk = segment[start_idx:end_idx]
                # RMS amplitude calculation
                amplitude = np.sqrt(np.mean(chunk**2)) * 2.0
                bar_heights[j] = amplitude * weighting_factor[j]
                
        batch_results.append(bar_heights)
        
    return batch_results

def smooth_heights(heights_list, window_size=3):
    """Apply temporal smoothing to bar heights between frames using vectorized operations."""
    if window_size <= 1:
        return heights_list
        
    # Convert to numpy array for faster operations
    heights_array = np.array(heights_list)
    smoothed = np.zeros_like(heights_array)
    half_window = window_size // 2
    
    for i in range(len(heights_array)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(heights_array), i + half_window + 1)
        
        # Create weight array (more weight to current frame)
        weights = np.ones(end_idx - start_idx)
        center_idx = i - start_idx
        if center_idx >= 0 and center_idx < len(weights):
            weights = weights * 0.5 / (len(weights) - 1)
            weights[center_idx] = 0.5
        else:
            weights = weights / len(weights)
            
        # Apply weights
        weighted_sum = np.sum(heights_array[start_idx:end_idx] * weights[:, np.newaxis], axis=0)
        smoothed[i] = weighted_sum
        
    return smoothed

def create_waveform_video(input_file, output_file=None, color='#00FF00', quality='medium', 
                          num_workers=None, max_duration=None):
    """
    Creates a minimalist waveform visualization video from an audio file
    with optimized performance.
    
    Parameters:
    -----------
    input_file  : path to input audio file
    output_file : path to output video file (optional)
    color       : waveform color (hex code or name)
    quality     : 'low', 'medium', or 'high' (affects FPS and resolution)
    num_workers : number of CPU cores to use (None = auto-detect)
    max_duration: maximum duration to process in seconds (None = auto-cap at 10 min)
    """
    # Generate output filename if not provided
    if output_file is None:
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_waveform.mp4"

    print(f"Processing audio file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Waveform color: {color}")
    
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
        fps = 24
        dpi = 90
        num_bars = 36
        smoothing_window = 3
    elif quality == 'high':
        fps = 60
        dpi = 150
        num_bars = 60
        smoothing_window = 5
    else:  # medium (default)
        fps = 30
        dpi = 120
        num_bars = 46
        smoothing_window = 5
    
    print(f"Quality: {quality} (FPS={fps}, DPI={dpi}, Bars={num_bars})")
    
    total_frames = int(fps * duration)
    samples_per_frame = int(sr / fps)
    
    # Create a weighting factor (Hann window)
    window = np.hanning(num_bars)
    weighting_factor = 0.5 + 0.5 * window
    
    # Pre-compute all bar heights using parallel processing
    print("Pre-computing waveform data using parallel processing...")
    
    batch_size = max(10, total_frames // (num_workers * 2))  # Aim for ~2 batches per worker
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare batches for parallel processing
        batch_args = []
        for batch in range(total_batches):
            start_frame = batch * batch_size
            end_frame = min((batch + 1) * batch_size, total_frames)
            batch_args.append((
                start_frame, end_frame, samples_per_frame, y, sr, num_bars, weighting_factor
            ))
        
        # Process batches in parallel
        results = list(executor.map(compute_frame_bars_batch, batch_args))
        
    # Flatten results
    all_bar_heights = []
    for batch_result in results:
        all_bar_heights.extend(batch_result)
    
    # Apply smoothing for a cleaner animation
    print("Applying temporal smoothing...")
    all_bar_heights = smooth_heights(all_bar_heights, window_size=smoothing_window)
    
    print("Finished pre-computing waveform data")
    
    # Set up the figure
    fig_width, fig_height = 16, 9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='black', dpi=dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Use a slightly wider spacing for fewer bars
    bar_x = np.arange(num_bars) * 1.5
    bar_width = 0.9
    
    # Pre-create rectangle objects
    up_rects = [plt.Rectangle((0, 0), 0, 0) for _ in range(num_bars)]
    down_rects = [plt.Rectangle((0, 0), 0, 0) for _ in range(num_bars)]
    
    # Add the rectangles to the axis once
    for rect in up_rects + down_rects:
        ax.add_patch(rect)
    
    # Fixed visualization parameters
    ax.set_xlim(bar_x[0] - 1, bar_x[-1] + 2)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    def animate(i):
        bar_heights = all_bar_heights[i]
        
        # Update rectangle positions and sizes
        for j, height in enumerate(bar_heights):
            x_left = bar_x[j] - bar_width/2
            # Update top rectangle
            up_rects[j].set_bounds(x_left, 0, bar_width, height)
            # Update bottom rectangle (mirrored)
            down_rects[j].set_bounds(x_left, -height, bar_width, height)
        
        # Set colors for all rectangles
        for rect in up_rects + down_rects:
            rect.set_color(color)
            rect.set_alpha(0.9)
        
        return []
    
    def make_frame(t):
        i = int(t * fps)
        if i >= total_frames:
            i = total_frames - 1
        animate(i)
        return mplfig_to_npimage(fig)
    
    print("Generating video frames...")
    animation_clip = mpe.VideoClip(make_frame, duration=duration)
    
    print("Adding audio to video...")
    audio_clip = mpe.AudioFileClip(input_file).subclip(0, duration)
    final_clip = animation_clip.set_audio(audio_clip)
    
    # Determine video bitrate based on quality
    if quality == 'low':
        bitrate = "4000k"
    elif quality == 'high':
        bitrate = "10000k"
    else:
        bitrate = "6000k"
    
    print(f"Writing video to {output_file}...")
    final_clip.write_videofile(
        output_file,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        bitrate=bitrate,
        threads=num_workers  # Use multiple threads for video encoding
    )
    
    plt.close(fig)
    print("Visualization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a minimalist waveform visualization video from audio"
    )
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("-o", "--output_file", help="Path to output MP4 file (optional)")
    parser.add_argument("-c", "--color", default="#00FF00",
                        help="Waveform color (hex code or color name, default: #00FF00)")
    parser.add_argument("-q", "--quality", default="medium", choices=["low", "medium", "high"],
                        help="Output quality (affects resolution and FPS)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of CPU cores to use (default: auto-detect)")
    parser.add_argument("-d", "--max_duration", type=float, default=None,
                        help="Maximum duration to process in seconds")
    
    args = parser.parse_args()
    create_waveform_video(
        args.input_file, 
        args.output_file, 
        args.color,
        args.quality,
        args.workers,
        args.max_duration
    )
