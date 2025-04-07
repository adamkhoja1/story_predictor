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

RESOLVE_QUESTIONS_PROMPT = """
# Short Story Excerpt

{text}

# Instructions

You have been provided with the entirety of a short story, and I will soon ask a question which someone asked about the story halfway through. In light of the rest of the story, you will resolve the question as "yes" or "no" or "ambiguous". The question you will consider is:
{question}

# Output Format

First, think through and surface the key elements of the story which are significant to understanding the story and how it relates to the question. Summarize events or observations which might be central to answering the question. Place this section of your response in <think></think> tags.

Second, resolve the question by responding with one of <answer>yes</answer>, <answer>no</answer>, or <answer>ambiguous</answer>, using the information provided in the story.
"""

def extract_answer_from_response(response_text: str) -> str:
    """
    Extract answer from model response using XML-style tags.
    
    Args:
        response_text (str): Raw response from the model
        
    Returns:
        str: extracted answer
    """
    # Find all content between <answer> and </answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, response_text, re.DOTALL)
    
    # Clean and validate tags
    if len(matches) != 1:
        raise ValueError("Trouble extracting a single answer from response")
    return matches[0].strip()

async def resolve_question(text: str, question: str) -> str:
    """
    Asynchronously resolve a question using Google's API.
    
    Args:
        excerpt (str): The story excerpt to create questions for
        num_questions (int): The number of questions to create
    
    Returns:
        str: extracted answer
    """
    prompt = RESOLVE_QUESTIONS_PROMPT.format(text=text, question=question)

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            # model="gemini-2.5-pro-exp-03-25",
            contents=prompt
        )
        
        return extract_answer_from_response(response.text)
        
    except Exception as e:
        print(f"Error resolving question: {e}")
        return "ambiguous"

async def resolve_question_from_story(metadata: dict, file_path: Path, question: str) -> str:
    """
    Asynchronously resolve a question from a story from a file.
    """
    with open(file_path, "r") as f:
        text = f.read()
    return await resolve_question(text, question)

async def main():
    with open("metadata/short_stories.json", "r") as f:
        metadata = json.load(f)
    with open("forecast_data/questions_0_440.json", "r") as f:
        questions = json.load(f)
        questions_dict = {q["id"]: q["questions"] for q in questions}
    
    start_idx = 10
    end_idx = 440
    metadata = metadata[start_idx:end_idx]
    
    # Limit concurrent tasks
    max_concurrent_tasks = 20
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_story(data, question):
        async with semaphore:
            answer = await resolve_question_from_story(data, Path(f"stories_cleaned/{data['id']}.txt"), question)
            return data, question, answer
    
    results = []
    
    # Create and gather all tasks
    tasks = [process_story(data, question) for data in metadata for question in questions_dict[data["id"]]]
    
    # Process results as they complete
    for future in asyncio.as_completed(tasks):
        data, story_question, answer = await future
        if answer != "ambiguous":
            results.append({
                "id": data["id"],
                "question": story_question,
                "answer": answer
            })
            print(f"Resolved {data['id']} with question {story_question}")
    
    # Consolidate results
    ret_results = {data["id"]: data for data in metadata}
    for result in results:
        if "questions" not in ret_results[result["id"]]:
            ret_results[result["id"]]["questions"] = []
        ret_results[result["id"]]["questions"].append({
            "question": result["question"],
            "answer": result["answer"]
        })


    with open(f"forecast_data/results_{start_idx}_{end_idx}.json", "w") as f:
        json.dump(ret_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())