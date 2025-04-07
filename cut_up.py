import re
from pathlib import Path
from tqdm import tqdm
import json

pattern = r"""This ebook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this ebook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: (.+)

Author: (.+)

Release date: (.+?)\[.*
.*

Language: (.+)"""

weak_pattern = r"""before using this eBook..(.+?)\*\*\*.START"""

def extract_metadata(text: str) -> dict:
    match = re.search(weak_pattern, text, re.DOTALL)
    fields = match.group(1).split("\n\n")
    fields = [x.split(":") for x in fields if x.strip()]
    fields = [(x[0].strip(), ":".join(x[1:]).replace("\n", " ").strip()) for x in fields]
    return {k:v for k,v in fields}

# def extract_metadata_2(text: str) -> dict:
#     lines = text.split('\n')
    
#     # Find the start and end of the metadata section
#     start_idx = None
#     end_idx = len(lines)
    
#     # Find start by looking for common header fields
#     for i, line in enumerate(lines):
#         stripped = line.strip()
#         if (stripped.startswith("Title:") or 
#             stripped.startswith("Author:") or 
#             stripped.startswith("Release date:")):
#             start_idx = i
#             break
    
#     if start_idx is None:
#         return metadata  # No metadata section found
    
#     # Find end by looking for the start of content marker
#     for i, line in enumerate(lines[start_idx:], start_idx):
#         if "*** START OF" in line:
#             end_idx

def trim_to_raw_text(text: str) -> tuple[str, int]:
    cut_up = text.split("***")[2:-2]
    text = "***".join(cut_up)
    newline_cut = text.split("\n\n")
    newline_cut = [x.replace("\n", " ") for x in newline_cut if x.strip()]
    return "\n\n".join(newline_cut), sum(len(x.split()) for x in newline_cut)

# def process_dir(dir: Path) -> bool:
#     txt_file = dir / f"pg{dir.name}.txt"
#     if not txt_file.exists():
#         print(f"File {txt_file} does not exist")
#         return False
#     with open(txt_file, "r") as f:
#         text = f.read()
#     match = re.search(lang_pattern, text)
#     if match:
#         lang = match.group(1)
#         if lang != "English":
#             return False
#     else:
#         print(f"File {txt_file} does not have a language")
#         return False
#     return True

def collect_metadata(dir: Path) -> dict:
    txt_file = dir / f"pg{dir.name}.txt"
    try:
        with open(txt_file, "r") as f:
            text = f.read()
        metadata = extract_metadata(text)
        _, wc = trim_to_raw_text(text)
    except Exception as e:
        print(f"Error reading file {txt_file}: {e}")
        return {
            "id": dir.name,
            "error": True
        }
    return {
        "id": dir.name,
        "error": False,
        **metadata,
        "word_count": wc
    }

def clean_and_filter_metadata(metadata: dict) -> dict:
    new_metadata = []
    for data in metadata:
        # Check if error
        if data["error"]:
            continue
        # Check word count < 8000
        if data["word_count"] > 8000:
            continue
        # Check language is English
        if data["Language"] != "English":
            continue
        # Get best guess for year
        year_pattern = r"\s(\d{4})"
        try:
            pub_string = data.get("Original publication", "") + data["Release date"]
            years = re.findall(year_pattern, pub_string)
            year = int(min(years))
        except Exception as e:
            print(f"Error getting year for {data['id']}: {e}")
            continue
        # Add to new metadata
        new_metadata.append(data | {"pub_year": year})
        new_metadata[-1].pop("Credits", "")
    return new_metadata

def main():
    # big_dir = Path("txt_files/")
    # metadata = []
    # for file in tqdm(big_dir.iterdir()):
    #     metadata.append(collect_metadata(file))
    # with open("metadata.json", "w") as f:
    #     json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open("metadata/metadata.json", "r") as f:
        existing_metadata = json.load(f)
    new_metadata = clean_and_filter_metadata(existing_metadata)
    with open("metadata/metadata_clean.json", "w") as f:
        json.dump(new_metadata, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()