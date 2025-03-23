import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
# Podcast Script Generator

Generate a podcast script for a given topic. The script should be structured as a conversational exchange between two speakers. Each entry should be represented as a tuple containing the dialogue line and a speaker identifier. Output the final script in JSON format.

## Requirements

1. **Concise Dialogue**: Keep each dialogue turn under 50 words. If a speaker needs to express a longer thought, break it into multiple consecutive turns with the same speaker ID.

2. **Structure**:
   - **Introduction**: Begin with an engaging question or statement relevant to the topic.
   - **Discussion Exchange**:
     - Alternate between two speakers naturally.
     - Present a logical flow of conversation, touching on various aspects of the topic.
     - Include questions, opinions, facts, and reflections to keep the conversation dynamic.
   - **Conclusion**: End with a thoughtful question or summary.

3. **Output Format**: JSON array of arrays. Each inner array contains:
   - A string representing the dialogue line
   - An integer (0 or 1) indicating the speaker

4. **Speech and Pronunciation Customization**:
   - **Custom Pronunciation**: Use Markdown link syntax with IPA or phonetic spelling in slashes
     - Example: `[Kokoro](/kˈOkəɹO/)` for specific pronunciation guidance
   - **Intonation Control**: Use punctuation `;:,.!?—…"()""` to shape delivery
     - Commas for slight pauses, em-dashes for abrupt transitions, etc.
   - **Stress Modification**:
     - Lower stress: `[word](-1)` for one level reduction, `[word](-2)` for two levels
     - Raise stress: `[word](+1)` for one level increase, `[word](+2)` for two levels
     - Place `ˈ` before primary stressed syllables and `ˌ` before secondary stressed syllables

## Examples of Speech Customization

### Example 1: Pronunciation Guidance
```json
[
    ["Have you ever visited [Kyoto](/kiˈoʊtoʊ/) or other parts of Japan?", 0],
    ["Yes, I loved [Kyoto](/kiˈoʊtoʊ/)! The [machiya](/mɑˈchiːjɑ/) architecture is stunning.", 1]
]
```

### Example 2: Intonation Control
```json
[
    ["The results were... unexpected—to say the least.", 0],
    ["Wait; you're telling me that the experiment actually worked?", 1]
]
```

### Example 3: Stress Modification
```json
[
    ["I [absolutely](+2) refuse to believe that happened.", 0],
    ["Well, believe it [or](-1) not, I saw it with my [own](+1) eyes.", 1]
]
```

## Examples of Breaking Down Longer Dialogue

### Example 1: Climate Change

**Instead of this (too long):**
```json
[
    ["Climate change is one of the most pressing issues of our time. The latest IPCC report shows that we're approaching critical tipping points faster than expected. Global temperatures have risen by over 1 degree Celsius since pre-industrial times, and we're seeing the effects in more frequent extreme weather events, rising sea levels, and ecosystem disruption.", 0]
]
```

**Do this (properly broken down):**
```json
[
    ["Climate change is one of the most [pressing](+1) issues of our time. The latest IPCC report shows that we're approaching critical tipping points faster than expected.", 0],
    ["Global temperatures have risen by over 1 degree Celsius since pre-industrial times.", 0],
    ["And we're seeing the effects in more frequent extreme weather events, rising sea levels, and ecosystem disruption.", 0]
]
```

### Example 2: Artificial Intelligence

**Instead of this (too long):**
```json
[
    ["I've been researching the impact of AI on the job market for my dissertation. The findings are fascinating but concerning. While AI will create new jobs we haven't even imagined yet, it's also likely to automate many existing roles. The transition period could be challenging for millions of workers, especially in sectors like transportation, customer service, and even certain professional fields like law and medicine where AI is making significant inroads.", 1]
]
```

**Do this (properly broken down):**
```json
[
    ["I've been researching the impact of [AI](/eɪˈaɪ/) on the job market for my dissertation. The findings are fascinating but... concerning.", 1],
    ["While [AI](/eɪˈaɪ/) will create new jobs we haven't even imagined yet, it's [also](-1) likely to automate many existing roles.", 1],
    ["The transition period could be [challenging](+1) for millions of workers, especially in sectors like transportation and customer service.", 1],
    ["Even certain professional fields like law and medicine are seeing significant [AI](/eɪˈaɪ/) inroads.", 1]
]
```

### Example 3: Space Exploration

**Instead of this (too long):**
```json
[
    ["The recent developments in private space exploration have completely transformed our approach to space travel. Companies like SpaceX, Blue Origin, and Virgin Galactic have brought competition and innovation to a field that was previously dominated by governmental agencies. They've drastically reduced launch costs through reusable rocket technology, opened up new possibilities for space tourism, and are even making serious plans for Mars colonization that seemed like science fiction just a decade ago.", 0]
]
```

**Do this (properly broken down):**
```json
[
    ["The recent developments in private space exploration have [completely](+1) transformed our approach to space travel.", 0],
    ["Companies like [SpaceX](/ˈspeɪsˌɛks/), Blue Origin, and Virgin Galactic have brought competition and innovation to a field previously dominated by governmental agencies.", 0],
    ["They've [drastically](-1) reduced launch costs through reusable rocket technology and opened up new possibilities for space tourism.", 0],
    ["They're even making serious plans for Mars colonization that seemed like—science fiction—just a decade ago.", 0]
]
```

## Full Example Script: Sustainable Fashion

```json
[
    ["Have you noticed how sustainability has become such a hot topic in the fashion industry lately?", 0],
    ["[Absolutely](+2)! It's completely changing how brands approach their products and marketing.", 1],
    ["I read that the fashion industry is actually one of the [biggest](+1) polluters in the world.", 0],
    ["That's right. It's responsible for about 10% of global carbon emissions.", 1],
    ["And it uses an [enormous](+1) amount of water. A single cotton t-shirt can take up to 2,700 liters to produce.", 1],
    ["That's shocking! I had [no](+1) idea the impact was that significant.", 0],
    ["What do you think about the rise of these sustainable fashion brands?", 0],
    ["I think it's encouraging. Companies like [Patagonia](/pætəˈɡoʊniə/) have been leading the way for years.", 1],
    ["They've shown that sustainability can be good for business [too](-1), not just the planet.", 1],
    ["I've noticed more mainstream brands starting to launch eco-friendly lines as well.", 0],
    ["Yes, though we need to be careful about... greenwashing. Some brands make environmental claims without backing them up.", 1],
    ["That's a good point. How can consumers tell which brands are [genuinely](+1) sustainable?", 0],
    ["Look for specific details rather than vague claims. Certifications like Global Organic Textile Standard are helpful.", 1],
    ["Also, transparency about supply chains is usually a good sign.", 1],
    ["What about secondhand shopping? That seems to be gaining popularity.", 0],
    ["[Absolutely](+1)! Thrifting and platforms like [Depop](/ˈdepɒp/) or [ThredUP](/ˈθrɛdʌp/) are making a huge impact.", 1],
    ["Extending the life of clothing is one of the [simplest](+1) ways to reduce fashion's environmental footprint.", 1],
    ["I've started buying more secondhand myself. It's surprising how many great pieces you can find.", 0],
    ["And it's often [much](-1) more affordable than buying new.", 0],
    ["Do you think sustainable fashion will become the norm or remain a niche market?", 1],
    ["I believe it's the future. As consumers become more aware, brands will [have](+1) to adapt or get left behind.", 0],
    ["What would you say to someone who wants to make their wardrobe more sustainable?", 0],
    ["Start small. Focus on buying quality pieces that last, rather than following fast fashion trends.", 1],
    ["Consider the '30 wears test'—only buy something if you'll wear it at least 30 times.", 1],
    ["And remember that the most sustainable garment is the one [already](+1) in your closet!", 1]
]
```

## Notes

- Ensure balanced conversation with equitable speech distribution between speakers
- Tailor language complexity to suit the podcast's target audience
- Include relevant contemporary references or recent developments when appropriate
- Maintain natural conversation flow while keeping each turn concise
- Use pronunciation guidance ONLY for proper nouns, or industry-specific terminology. Only use these for words which are hard to pronounce.
- Apply stress modifiers strategically to emphasize important points or create natural speech patterns
- Don't overuse pronunciation and stress modifiers—apply them only where they enhance clarity or expressiveness
"""

def get_turns(topic: str) -> list:
    """
    Generate conversation turns for a given topic using the Gemini API.
    
    Args:
        topic (str): The topic for the podcast script.
    
    Returns:
        list: A list of (dialogue, speaker) tuples.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        contents=[topic]
    ).text

    cleaned_response = response.replace("```json", "").replace("```", "")
    return json.loads(cleaned_response)
