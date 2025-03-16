import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import moviepy.editor as mpe
from moviepy.video.io.bindings import mplfig_to_npimage
import argparse
import os

def compute_frame_bars(frame_data, weighting_factor):
    """
    Compute the height of each bar for a given animation frame.
    
    Parameters:
    -----------
    frame_data      : (frame_idx, samples_per_frame, y, sr, num_bars)
    weighting_factor: numpy array of length num_bars (e.g., Hann window)
    
    Returns:
    --------
    bar_heights: list of bar heights (length = num_bars)
    """
    frame_idx, samples_per_frame, y, sr, num_bars = frame_data
    start_sample = frame_idx * samples_per_frame
    end_sample = min(start_sample + samples_per_frame, len(y))
    segment = y[start_sample:end_sample]
    
    samples_per_bar = max(1, len(segment) // num_bars)
    bar_heights = []
    
    for j in range(num_bars):
        start_idx = j * samples_per_bar
        end_idx = min((j+1) * samples_per_bar, len(segment))
        if start_idx < len(segment):
            chunk = segment[start_idx:end_idx]
            # Base amplitude calculation
            amplitude = np.sqrt(np.mean(chunk**2)) * 2.0
            # Apply the weighting factor so bars near edges are smaller
            height = amplitude * weighting_factor[j]
            bar_heights.append(height)
        else:
            bar_heights.append(0)
            
    return bar_heights

def create_waveform_video(input_file, output_file=None, color='#00FF00'):
    """
    Creates a minimalist waveform visualization video from an audio file
    with a Hann-window-based weighting factor so the bars at the edges
    are shorter, gradually increasing toward the center.
    """
    # Generate output filename if not provided
    if output_file is None:
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_waveform.mp4"

    print(f"Processing audio file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Waveform color: {color}")
    
    # Load the audio file
    print("Loading audio file...")
    y, sr = librosa.load(input_file, sr=None, mono=True)
    
    # Get the duration of the audio
    audio_duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Choose an FPS based on duration
    if audio_duration <= 30:
        fps = 60
    elif audio_duration <= 120:
        fps = 45
    else:
        fps = 30
    
    # Cap visualization to 10 minutes if needed
    if audio_duration > 600:
        duration = 600
        print(f"Audio is very long ({audio_duration:.2f}s). Capping visualization to first 10 minutes.")
    else:
        duration = audio_duration
    
    dpi = 120
    print(f"Auto-configured settings: FPS={fps}, DPI={dpi}, Duration={duration:.2f}s")
    
    total_frames = int(fps * duration)
    samples_per_frame = int(sr / fps)
    
    # Reduced number of bars for a sparser look
    num_bars = 46
    
    # ----------------------------
    # Create a weighting factor
    # ----------------------------
    # Example: Hann window for a smooth "bell-shaped" amplitude weighting,
    # scaled so edges are ~0.5, center is ~1.0
    window = np.hanning(num_bars)          # 0 at edges, 1 in middle
    weighting_factor = 0.5 + 0.5 * window  # edges -> 0.5, center -> 1.0
    
    # Pre-compute all bar heights to speed up animation
    print("Pre-computing waveform data...")
    
    def smooth_heights(heights_list, window_size=3):
        """Apply simple temporal smoothing to bar heights between frames."""
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(heights_list)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(heights_list), i + half_window + 1)
            
            # Weighted average: more weight to the current frame
            weights = np.array([
                0.5 if j == i else 0.5/(end_idx-start_idx-1) 
                for j in range(start_idx, end_idx)
            ])
            weighted_sum = np.zeros_like(heights_list[i])
            weight_sum = 0
            
            for j, w in zip(range(start_idx, end_idx), weights):
                weighted_sum += np.array(heights_list[j]) * w
                weight_sum += w
                
            smoothed.append(weighted_sum / weight_sum)
        return smoothed
    
    all_bar_heights = []
    
    batch_size = 100
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    for batch in range(total_batches):
        start_frame = batch * batch_size
        end_frame = min((batch + 1) * batch_size, total_frames)
        
        for frame_idx in range(start_frame, end_frame):
            frame_data = (frame_idx, samples_per_frame, y, sr, num_bars)
            bar_heights = compute_frame_bars(frame_data, weighting_factor)
            all_bar_heights.append(bar_heights)
        
        print(f"Processed {end_frame}/{total_frames} frames "
              f"({(end_frame/total_frames*100):.1f}%)")
    
    # Apply smoothing for a cleaner animation
    print("Applying temporal smoothing...")
    all_bar_heights = smooth_heights(all_bar_heights, window_size=5)
    
    print("Finished pre-computing waveform data")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='black', dpi=dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Fewer bars, spaced out manually
    bar_x = np.arange(num_bars) * 1.5
    bar_width = 0.9
    
    def animate(i):
        ax.clear()
        ax.set_facecolor('black')
        
        # X-limits around the bars
        ax.set_xlim(bar_x[0] - 1, bar_x[-1] + 2)
        ax.set_ylim(-1.1, 1.1)
        
        # Remove all ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        bar_heights = all_bar_heights[i]
        
        up_rects = []
        down_rects = []
        
        for j, height in enumerate(bar_heights):
            x_left = bar_x[j] - bar_width/2
            # top half
            up_rects.append((x_left, 0, bar_width, height))
            # mirrored bottom half
            down_rects.append((x_left, -height, bar_width, height))
        
        up_collection = matplotlib.collections.PatchCollection(
            [plt.Rectangle((x, y), w, h) for x, y, w, h in up_rects],
            color=color, alpha=0.9
        )
        down_collection = matplotlib.collections.PatchCollection(
            [plt.Rectangle((x, y), w, h) for x, y, w, h in down_rects],
            color=color, alpha=0.9
        )
        
        ax.add_collection(up_collection)
        ax.add_collection(down_collection)
        
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
    
    print(f"Writing video to {output_file}...")
    final_clip.write_videofile(
        output_file,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        bitrate="8000k"
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
    
    args = parser.parse_args()
    create_waveform_video(args.input_file, args.output_file, args.color)
