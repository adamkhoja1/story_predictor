import os
from pathlib import Path
import asyncio
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import re
import json
from cut_up import trim_to_raw_text

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Example tags
TAGS = [
    "mystery",
    "romance",
    "historical",
    "horror",
    "fantasy",
    "science fiction",
    "literary",
    "adventure",
    "humor",
    "allegorical",
    "satire",
    "tragedy",
    "comedy",
    "drama",
    "not a short story"
]

def extract_tags_from_response(response_text: str) -> list[str]:
    """
    Extract tags from model response using XML-style tags.
    
    Args:
        response_text (str): Raw response from the model
        
    Returns:
        list[str]: list of extracted tags
    """
    # Find all content between <tag> and </tag>
    tag_pattern = r'<tag>(.*?)</tag>'
    matches = re.findall(tag_pattern, response_text, re.IGNORECASE)
    
    # Clean and validate tags
    tags = [tag.strip().lower() for tag in matches]
    valid_tags = [tag for tag in tags if tag in TAGS]
    
    return valid_tags

async def check_short_story_async(title: str) -> bool:
    """
    Check if a story is likely to be a short story based on its title.
    
    Args:
        title (str): The story title
        
    Returns:
        bool: True if likely a short story, False otherwise
    """
    prompt = f"""Analyze this publication title and determine if it's likely to be a short story:
"{title}"

Respond with <tag>not a short story</tag> if your knowledge of the title or its content suggests this is NOT a short story (e.g., if it's a novel, speech, poem, collection, non-fiction work, etc.)
Respond with <tag>short story</tag> if it appears to be a short story OR if you're unsure (be conservative in labeling as "not a short story" if there is uncertainty).

Only respond with the appropriate tag, nothing else."""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        tags = extract_tags_from_response(response.text)
        return "not a short story" not in tags
        
    except Exception as e:
        print(f"Error checking title: {e}")
        return True  # Default to processing the story if we can't check the title

async def tag_story_text(text: str, TAGS: list[str]) -> list[str]:
    """
    Asynchronously tag a story using Google's API based on a set of available tags.
    
    Args:
        text (str): The story text to tag
        TAGS (list[str]): list of available tags to choose from
    
    Returns:
        list[str]: list of tags that were assigned to the story
    """
    prompt = f"""# Instructions

Please analyze the following story and tag it with the most relevant tags from this list: {', '.join(TAGS)}
Your response should be in XML format like this: <tag>tag1</tag><tag>tag2</tag>
Respond with ONLY <tag>not a short story</tag> if the story is NOT a short story (e.g., if it's a novel, speech, poem, collection, non-fiction work, etc.)
If the story is a short story, respond with up to 3 tags from the list that closely describe the story.

# Story (possibly truncated)

{text[:10000]}

# Tag the Story

Please respond with only the relevant tags in XML format like this: <tag>tag1</tag><tag>tag2</tag>
Importantly, if the text is not a short story, respond with <tag>not a short story</tag> and nothing else. In cases where you are unsure, respond with <tag>not a short story</tag>.
Do not include any other text or explanation."""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return extract_tags_from_response(response.text)
        
    except Exception as e:
        print(f"Error tagging story: {e}")
        return []

async def tag_story(metadata: dict, file_path: Path, TAGS: list[str]) -> list[str]:
    """
    Asynchronously tag a story from a file using Google's API. First check if the story is a short story from the title, then tag the story.
    """
    if await check_short_story_async(metadata["Title"]):
        with open(file_path, "r") as f:
            text = f.read()
        text, _ = trim_to_raw_text(text)
        tags = await tag_story_text(text, TAGS)
        return tags if tags != [] and "not a short story" not in tags else []
    return []

async def main():
    with open("metadata/metadata_clean.json", "r") as f:
        metadata = json.load(f)
    
    start_idx = 0
    end_idx = 1100
    metadata = metadata[start_idx:end_idx]
    
    # Limit concurrent tasks
    max_concurrent_tasks = 20
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_story(data):
        async with semaphore:
            tags = await tag_story(data, Path(f"txt_files/{data['id']}/pg{data['id']}.txt"), TAGS)
            return data, tags
    
    new_metadata = []
    
    # Create and gather all tasks
    tasks = [process_story(data) for data in metadata]
    
    # Process results as they complete
    for future in asyncio.as_completed(tasks):
        data, tags = await future
        if tags != []:
            data["tags"] = tags
            new_metadata.append(data)
            print(f"Tagged {data['id']} with {tags}")
    
    with open(f"metadata/short_story_metadata_{start_idx}_{end_idx}.json", "w") as f:
        json.dump(new_metadata, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())