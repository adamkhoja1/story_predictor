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

CREATE_QUESTIONS_PROMPT = """
# Short Story Excerpt

{excerpt}

# Instructions

You have been provided with the first half of a short story, and will be tasked with producing a variety of specific, objective binary forecasting questions about what will happen later in the story. Your questions might primarily pertain to specific plot outcomes, but a few might also reference flavors where applicable such as:
- How the relationships between characters change over the course of the story
- Setting and time
- Important thematic or symbolic events or imagery

# Examples of Questions

Here are some examples of good questions for an unrelated story about a Knight named Sir Example (story not rendered here):
1. <question>Will Sir Example eventually receive his promised audience with the King?</question>
2. <question>Will we discover who sent Sir Example the threatening letter by the end of the story?</question>
3. <question>Was Count Wigfrid the person who sent Sir Example the threatening letter?</question>
4. <question>Will more than a total of 3 of Sir Example's fellow knights be slain by the end of the story?</question>
5. <question>Will the symbolism of the thrush sitting on Sir Example's helmet occur or be mentioned at least twice in the story?</question>
6. <question>Will the Rock Golems that The Count is working with remain faithful to his cause?</question>
7. <question>Will the story end before the aforementioned Festival of the Sun?</question>
8. <question>Will Sir Knight return to his home village by the end of the story?</question>

# Notes and Rules

- Questions should not be relative to a particular point in the story, in the sense that one should be able to answer the question if they have the full story but don't know where you had read to when you asked it. An example question which is undesirable for this reason might be "Does Sir Example next visit the town of Dade?"
- Though any question you formulate might not be answered unambiguously by the end of the stories, lean toward asking questions which seem somewhat likely to be answered by the end of the story.
- Aim for questions which are substantive and speak to the main plot and the story's key thematic elements, as opposed to tangential or not thematically important.
- Questions normally shouldn't make low-probability, speculative conjectures about what occurs in the story. However, a handful of such questions can be appropriate where thematically appropriate: for example, asking about the most plausible whodunnit in a mystery story, or whether a particularly obvious Chekhov's gun will fire by the end of a story.

# Output Format

First, think through and surface the key elements of the story which are significant and/or remain unresolved. Summarize events or observations which might be central to the story and relevant to the forecast. Place this section of your response in <think></think> tags.

Second, formulate {num_questions} questions for the story excerpt above. Format this as a numbered list with each question going in <question></question> XML tags.
"""

def extract_questions_from_response(response_text: str) -> list[str]:
    """
    Extract questions from model response using XML-style tags.
    
    Args:
        response_text (str): Raw response from the model
        
    Returns:
        list[str]: list of extracted questions
    """
    # Find all content between <question> and </question>
    question_pattern = r'<question>(.*?)</question>'
    matches = re.findall(question_pattern, response_text, re.DOTALL)
    
    # Clean and validate tags
    return [question.strip() for question in matches]

async def create_questions(excerpt: str, num_questions: int) -> list[str]:
    """
    Asynchronously create questions using Google's API.
    
    Args:
        excerpt (str): The story excerpt to create questions for
        num_questions (int): The number of questions to create
    
    Returns:
        list[str]: list of questions that were created
    """
    prompt = CREATE_QUESTIONS_PROMPT.format(excerpt=excerpt, num_questions=num_questions)

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return extract_questions_from_response(response.text)
        
    except Exception as e:
        print(f"Error creating questions: {e}")
        return []

async def operationalize_story(metadata: dict, file_path: Path, num_questions: int) -> list[str]:
    """
    Asynchronously create questions from a story from a file.
    """
    with open(file_path, "r") as f:
        text = f.read()
    lines = text.split("\n\n")
    half_story = "\n\n".join(lines[:len(lines)//2])
    return await create_questions(half_story, num_questions)

async def main():
    with open("metadata/short_stories.json", "r") as f:
        metadata = json.load(f)
    
    start_idx = 0
    end_idx = 440
    metadata = metadata[start_idx:end_idx]
    
    # Limit concurrent tasks
    max_concurrent_tasks = 20
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_story(data):
        async with semaphore:
            questions = await operationalize_story(data, Path(f"stories_cleaned/{data['id']}.txt"), 12)
            return data, questions
    
    questions = []
    
    # Create and gather all tasks
    tasks = [process_story(data) for data in metadata]
    
    # Process results as they complete
    for future in asyncio.as_completed(tasks):
        data, story_questions = await future
        if story_questions != []:
            questions.append({
                "id": data["id"],
                "questions": story_questions
            })
            print(f"Operationalized {data['id']} with example question {story_questions[0]}")
    
    with open(f"forecast_data/questions_{start_idx}_{end_idx}.json", "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())