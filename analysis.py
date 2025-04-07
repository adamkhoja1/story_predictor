import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import math

# Function to calculate log score
def calculate_log_score(prediction, actual):
    """
    Calculate log score for a prediction.
    For binary yes/no questions:
    - If actual is 'yes', use the prediction probability
    - If actual is 'no', use 1 - prediction probability
    """
    if actual == "yes":
        # Clip to avoid log(0)
        return math.log(max(prediction, 0.01))
    else:  # actual is "no"
        return math.log(max(1 - prediction, 0.01))

# Load data
def load_data():
    # Load ground truth data
    with open("forecast_data/results.json", "r") as f:
        results = json.load(f)
    
    # Load model predictions
    with open("forecast_data/forecasts_gemini-2.0-flash_0_440.json", "r") as f:
        flash_forecasts = json.load(f)
    
    with open("forecast_data/forecasts_gemini-2.0-flash-lite_0_440.json", "r") as f:
        flash_lite_forecasts = json.load(f)
    
    return results, flash_forecasts, flash_lite_forecasts

def analyze_predictions(results, flash_forecasts, flash_lite_forecasts):
    # Create structure to store all log scores
    all_scores = {
        "flash": {},
        "flash_lite": {}
    }
    
    # Create structure for story-level aggregation
    story_scores = defaultdict(list)
    story_titles = {}
    story_tags = {}
    
    # Process each story
    for story_id, story_data in results.items():
        if "error" in story_data and story_data["error"]:
            continue
        
        # Store story metadata
        story_titles[story_id] = story_data.get("Title", f"Story {story_id}")
        story_tags[story_id] = story_data.get("tags", [])
        
        # Get questions for this story
        questions = story_data.get("questions", [])
        
        # Check if this story has predictions from both models
        if story_id not in flash_forecasts or story_id not in flash_lite_forecasts:
            continue
        
        # Process each question
        for question_idx, question_data in enumerate(questions):
            question_idx_str = str(question_idx)
            actual_answer = question_data["answer"]
            
            # Process flash model
            if question_idx_str in flash_forecasts[story_id]:
                prediction = flash_forecasts[story_id][question_idx_str]
                log_score = calculate_log_score(prediction, actual_answer)
                
                # Store in all_scores structure
                if story_id not in all_scores["flash"]:
                    all_scores["flash"][story_id] = {}
                all_scores["flash"][story_id][question_idx_str] = log_score
                
                # Store for story-level aggregation
                story_scores[story_id].append(log_score)
            
            # Process flash-lite model
            if question_idx_str in flash_lite_forecasts[story_id]:
                prediction = flash_lite_forecasts[story_id][question_idx_str]
                log_score = calculate_log_score(prediction, actual_answer)
                
                # Store in all_scores structure
                if story_id not in all_scores["flash_lite"]:
                    all_scores["flash_lite"][story_id] = {}
                all_scores["flash_lite"][story_id][question_idx_str] = log_score
    
    # Calculate average log scores by story
    story_avg_scores = {story_id: np.mean(scores) for story_id, scores in story_scores.items()}
    
    # Calculate overall model averages
    flash_all_scores = [score for story_scores in all_scores["flash"].values() for score in story_scores.values()]
    flash_lite_all_scores = [score for story_scores in all_scores["flash_lite"].values() for score in story_scores.values()]
    
    model_avg_scores = {
        "flash": np.mean(flash_all_scores),
        "flash_lite": np.mean(flash_lite_all_scores)
    }
    
    # Calculate average scores by tag
    tag_scores = defaultdict(list)
    for story_id, avg_score in story_avg_scores.items():
        tags = story_tags.get(story_id, [])
        for tag in tags:
            tag_scores[tag].append(avg_score)
    
    tag_avg_scores = {tag: np.mean(scores) for tag, scores in tag_scores.items()}
    
    return all_scores, story_avg_scores, model_avg_scores, tag_avg_scores, story_titles

def create_visualizations(story_avg_scores, model_avg_scores, tag_avg_scores, story_titles):
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")
    
    # 1. Plot overall model average scores
    plt.figure(figsize=(10, 6))
    models = list(model_avg_scores.keys())
    scores = [model_avg_scores[model] for model in models]
    
    plt.bar(models, scores, color=['#1f77b4', '#ff7f0e'])
    plt.title("Average Log Score by Model", fontsize=16)
    plt.ylabel("Average Log Score", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_avg_scores.png")
    
    # 2. Distribution of stories by log score
    plt.figure(figsize=(12, 8))
    story_ids = list(story_avg_scores.keys())
    scores = list(story_avg_scores.values())
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    sorted_story_ids = [story_ids[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_titles = [story_titles[story_id] for story_id in sorted_story_ids]
    
    # Find most and least predictable stories
    most_predictable_idx = sorted_indices[-1]
    least_predictable_idx = sorted_indices[0]
    most_predictable = story_titles[story_ids[most_predictable_idx]]
    least_predictable = story_titles[story_ids[least_predictable_idx]]
    
    # Plot histogram
    sns.histplot(scores, kde=True, bins=15)
    plt.title("Distribution of Stories by Log Score", fontsize=16)
    plt.xlabel("Average Log Score", fontsize=14)
    plt.ylabel("Number of Stories", fontsize=14)
    plt.tight_layout()
    plt.savefig("story_score_distribution.png")
    
    # 3. Bar chart of story scores with most and least predictable highlighted
    plt.figure(figsize=(14, 10))
    colors = ['#1f77b4'] * len(sorted_titles)
    colors[0] = '#d62728'  # Highlight least predictable
    colors[-1] = '#2ca02c'  # Highlight most predictable
    
    plt.barh(sorted_titles, sorted_scores, color=colors)
    plt.xlabel("Average Log Score", fontsize=14)
    plt.title("Average Log Score by Story (gemini-2.0-flash)", fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add annotations for most and least predictable
    plt.annotate(f"Most predictable: {most_predictable}", 
                xy=(0.98, 0.02), 
                xycoords='figure fraction',
                ha='right',
                color='#2ca02c',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#2ca02c', alpha=0.8))
    
    plt.annotate(f"Least predictable: {least_predictable}", 
                xy=(0.98, 0.06), 
                xycoords='figure fraction',
                ha='right',
                color='#d62728',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#d62728', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("story_scores.png")
    
    # 4. Average log score by tag
    plt.figure(figsize=(12, 8))
    tags = list(tag_avg_scores.keys())
    tag_scores = [tag_avg_scores[tag] for tag in tags]
    
    # Sort by score
    sorted_indices = np.argsort(tag_scores)
    sorted_tags = [tags[i] for i in sorted_indices]
    sorted_tag_scores = [tag_scores[i] for i in sorted_indices]
    
    plt.barh(sorted_tags, sorted_tag_scores, color='#1f77b4')
    plt.title("Average Log Score by Story Tag", fontsize=16)
    plt.xlabel("Average Log Score", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tag_scores.png")

def main():
    # Load data
    results, flash_forecasts, flash_lite_forecasts = load_data()
    
    # Analyze predictions
    all_scores, story_avg_scores, model_avg_scores, tag_avg_scores, story_titles = analyze_predictions(
        results, flash_forecasts, flash_lite_forecasts
    )
    
    # Print summary statistics
    print("\n===== PREDICTION ANALYSIS RESULTS =====\n")
    
    print("Average Log Score by Model:")
    for model, score in model_avg_scores.items():
        print(f"  {model}: {score:.4f}")
    
    print("\nTop 5 Most Predictable Stories:")
    top_stories = sorted(story_avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for story_id, score in top_stories:
        print(f"  {story_titles[story_id]}: {score:.4f}")
    
    print("\nTop 5 Least Predictable Stories:")
    bottom_stories = sorted(story_avg_scores.items(), key=lambda x: x[1])[:5]
    for story_id, score in bottom_stories:
        print(f"  {story_titles[story_id]}: {score:.4f}")
    
    print("\nAverage Log Score by Tag:")
    for tag, score in sorted(tag_avg_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag}: {score:.4f}")
    
    # Create visualizations
    create_visualizations(story_avg_scores, model_avg_scores, tag_avg_scores, story_titles)
    
    print("\nAnalysis complete. Visualizations saved.")

def main_2():
    # Load data
    results, flash_forecasts, flash_lite_forecasts = load_data()
    
    # Analyze predictions
    all_scores, story_avg_scores, model_avg_scores, tag_avg_scores, story_titles = analyze_predictions(
        results, flash_forecasts, flash_lite_forecasts
    )
    
    # Sort stories by their log scores
    sorted_stories = sorted(story_avg_scores.items(), key=lambda x: x[1])
    
    # Get 5 least predictable stories (lowest log scores)
    print("\n===== 5 LEAST PREDICTABLE STORIES (gemini-2.0-flash) =====")
    print(f"{'Story ID':<10} {'Log Score':<10} {'Story Title'}")
    print("-" * 60)
    for story_id, score in sorted_stories[:5]:
        print(f"{story_id:<10} {score:<10.4f} {story_titles[story_id]}")
    
    # Get 5 most predictable stories (highest log scores)
    print("\n===== 5 MOST PREDICTABLE STORIES (gemini-2.0-flash) =====")
    print(f"{'Story ID':<10} {'Log Score':<10} {'Story Title'}")
    print("-" * 60)
    for story_id, score in sorted_stories[-5::][::-1]:  # Reverse to get highest first
        print(f"{story_id:<10} {score:<10.4f} {story_titles[story_id]}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()
    main_2()