import argparse
import re
from podcast_agent.generator import get_turns
from podcast_agent.audio import generate_audio

def main():
    parser = argparse.ArgumentParser(
        description="Generate a podcast on a given topic using Kokoro."
    )
    parser.add_argument("topic", type=str, help="The topic for the podcast")
    args = parser.parse_args()
    
    topic = args.topic
    print(f"Generating podcast for topic: {topic}")
    
    turns = get_turns(topic)

    filename_topic = re.sub(r"\s+", "_", topic)
    output_file = f"out/{filename_topic}_kokoro.wav"
    
    final_audio = generate_audio(turns, output_file)
    if final_audio:
        print(f"Audio generated successfully: {final_audio}")
    else:
        print("Audio generation failed.")

if __name__ == "__main__":
    main()

