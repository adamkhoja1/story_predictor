import os
import random
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
# Story Excerpt

{excerpt}

# Background

You will provide calibrated probabilistic forecasts for binary questions related to the story above, with access to only the first half of the story, with your performance evaluated according to the log scoring rule. When forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities, or 90% (9:1) and 99% (99:1) as similarly "high" probabilities. As the odds show, they are markedly different, so output your probabilities accordingly.

# Question to Forecast

{question}

Instructions:

1. Compress key background and events from the story, into a list of core factual points to reference. You might also wish to surface background knowledge about the tropes of similar stories. For this step, do not draw any conclusions about how a fact will influence your answer or forecast. Place this section of your response in <facts></facts> tags.
2. Provide a few reasons why the answer might be no. Rate the strength of each reason on a scale of 1-10. Use <no></no> tags.
3. Provide a few reasons why the answer might be yes. Rate the strength of each reason on a scale of 1-10. Use <yes></yes> tags.
4. Aggregate your considerations. Do not summarize or repeat previous points; instead, investigate how the competing factors and mechanisms interact and weigh against each other. Factorize your thinking across (exhaustive, mutually exclusive) cases if and only if it would be beneficial to your reasoning. Draw your reasoning both from the substance of the text, and from your wider intuitions about literature and story structure. Think like a superforecaster and a literary critic in one. Use <thinking></thinking> tags for this section of your response.
5. Output your final prediction (an integer percentage) in <answer></answer> tags. Your answer should match the regex r"<answer>\d\d?\%<\/answer>"
"""

def extract_probability_from_response(response_text: str) -> str:
    """
    Extract probability from model response using XML-style tags.
    
    Args:
        response_text (str): Raw response from the model
        
    Returns:
        str: extracted probability
    """
    # Find all content between <answer> and </answer>
    answer_pattern = r'<answer>(\d\d?)\%<\/answer>'
    matches = re.findall(answer_pattern, response_text, re.DOTALL)
    
    # Clean and validate tags
    if len(matches) != 1:
        raise ValueError("Trouble extracting a single probability from response")
    return float(matches[0])/100.0

async def forecast_question(excerpt: str, question: str, model: str = "gemini-2.0-flash") -> str:
    """
    Asynchronously forecast a question using Google's API.
    
    Args:
        excerpt (str): The story excerpt to create questions for
        num_questions (int): The number of questions to create
    
    Returns:
        str: extracted answer
    """
    prompt = RESOLVE_QUESTIONS_PROMPT.format(excerpt=excerpt, question=question)

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt
        )
        # For some proportion of generations, log the response
        if random.random() < 0.003:
            with open(f"forecast_data/example_forecasts_{model}.json", "a") as f:
                json.dump({
                    "question": question,
                    "response": response.text
                }, f, indent=2, ensure_ascii=False)
                f.write("\n")

        return extract_probability_from_response(response.text)
        
    except Exception as e:
        print(f"Error forecasting question: {e}")
        return None

async def forecast_question_from_story(metadata: dict, file_path: Path, question_idx: int, model: str = "gemini-2.0-flash") -> str:
    """
    Asynchronously forecast a question from the first half of a story from a file.
    """
    with open(file_path, "r") as f:
        text = f.read()
    lines = text.split("\n\n")
    half_story = "\n\n".join(lines[:len(lines)//2])
    return await forecast_question(half_story, metadata["questions"][question_idx]["question"], model)

async def main():
    with open("forecast_data/results.json", "r") as f:
        metadata = json.load(f)
    
    # model = "gemini-2.0-flash"
    model = "gemini-2.0-flash-lite"

    start_idx = 0
    end_idx = 440
    metadata = {k:v for k,v in list(metadata.items())[start_idx:end_idx]}

    # with open("forecast_data/forecasts_gemini-2.0-flash-lite_0_440.json", "r") as f:
    #     existing_forecasts = json.load(f)
    existing_forecasts = {}
    
    # Limit concurrent tasks
    max_concurrent_tasks = 50
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_story(data, question_idx):
        async with semaphore:
            answer = await forecast_question_from_story(data, Path(f"stories_cleaned/{data['id']}.txt"), question_idx, model)
            return data["id"], question_idx, answer
    
    forecasts = []
    
    # Create and gather all tasks
    # tasks = [process_story(data, question_idx) for data in metadata.values() for question_idx in range(len(data["questions"]))]
    tasks = []
    for data in metadata.values():
        for question_idx in range(len(data["questions"])):
            if str(question_idx) not in existing_forecasts.get(data["id"], {}):
                tasks.append(process_story(data, question_idx))

    # Process results as they complete
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        data_id, question_idx, answer = await future
        if answer is not None:
            forecasts.append({
                "id": data_id,
                "question_idx": question_idx,
                "answer": answer
            })
            print(f"Forecasted {data_id} with question {question_idx} - {answer:.2f}")

    # Consolidate forecasts by story
    # forecasts_by_story = {}
    forecasts_by_story = existing_forecasts
    for forecast in forecasts:
        if forecast["id"] not in forecasts_by_story:
            forecasts_by_story[forecast["id"]] = {}
        forecasts_by_story[forecast["id"]][forecast["question_idx"]] = forecast["answer"]

    with open(f"forecast_data/forecasts_{model}_{start_idx}_{end_idx}.json", "w") as f:
        json.dump(forecasts_by_story, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())