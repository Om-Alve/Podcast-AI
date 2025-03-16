import os
import re
import numpy as np
import soundfile as sf
from kokoro import KPipeline

def initialize_pipelines():
    """
    Initialize Kokoro pipelines for two speakers with distinct voices.
    
    Returns:
        tuple: (pipeline_speaker0, pipeline_speaker1, voices dict)
    """
    pipeline_speaker0 = KPipeline(lang_code='a')
    pipeline_speaker1 = KPipeline(lang_code='a')
    
    voices = {
        0: "am_fenrir",  # Voice for speaker 0
        1: "af_heart"     # Voice for speaker 1
    }
    
    return pipeline_speaker0, pipeline_speaker1, voices

def save_audio(audio_data, filename, sample_rate=24000):
    """
    Save audio data to a WAV file.
    
    Args:
        audio_data (np.ndarray): Audio data to save.
        filename (str): Path to the output file.
        sample_rate (int): Audio sample rate (default 24000).
    
    Returns:
        np.ndarray: The same audio data.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio_data, sample_rate)
    return audio_data

def generate_audio(turns, output_filename="out/full_conversation.wav"):
    """
    Generate and save audio for each conversation turn, then concatenate them.
    
    Args:
        turns (list of tuple): A list of tuples (text, speaker).
        output_filename (str): Path to the final concatenated audio file.
    
    Returns:
        str or None: The path to the final audio file or None if no audio was generated.
    """
    pipeline_speaker0, pipeline_speaker1, voices = initialize_pipelines()
    all_audio_data = []
    
    for i, (text, speaker) in enumerate(turns):
        print(f"Generating audio for turn {i} (Speaker {speaker}): {text[:30]}...")
        pipeline = pipeline_speaker0 if speaker == 0 else pipeline_speaker1
        voice = voices[speaker]
        paragraphs = re.split(r'\n+', text)
        turn_audio_data = []
        
        # Process each paragraph separately
        for paragraph in paragraphs:
            if paragraph.strip():
                generator = pipeline(paragraph, voice=voice, speed=1.0)
                for _, _, audio in generator:
                    turn_audio_data.append(audio)
                    
        if turn_audio_data:
            turn_audio = np.concatenate(turn_audio_data)
            # Add a pause of 0.5 seconds between turns
            pause = np.zeros(int(0.5 * 24000))
            all_audio_data.append(turn_audio)
            all_audio_data.append(pause)
    
    if all_audio_data:
        final_audio = np.concatenate(all_audio_data)
        save_audio(final_audio, output_filename)
        print(f"Final audio saved as {output_filename}")
        return output_filename
    else:
        print("No audio was generated!")
        return None

