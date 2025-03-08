import requests
import json
import time
from pydantic import BaseModel
from openai import OpenAI

# Define your OpenAI API key

api_key="API_KEY"
client = OpenAI(api_key = api_key)


# Set up the OpenAI API endpoint and headers
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Define the schema using Pydantic
class PromptMetadata(BaseModel):
    tags: list[str]
    negative_tags: list[str]
    lighting: str
    mood: str

# Function to fetch structured metadata for each prompt
def fetch_metadata(prompt):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that generates structured metadata for image prompts."},
                {"role": "user", "content": f"Please generate metadata for this prompt: \"{prompt}\". Structure it with keys 'tags', 'negative_tags', 'lighting', and 'mood'."}
            ],
            response_format=PromptMetadata
        )
        
        # Retrieve the parsed result
        metadata = completion.choices[0].message.parsed
        print(f"Metadata for prompt: {metadata}")
        return metadata
    except Exception as e:
        print(f"Error processing prompt: {prompt}")
        print(f"Error message: {e}")
        return None

# Load the JSON file with prompts
with open("dataset.json", "r") as file:
    dataset = json.load(file)

# Process each prompt in the dataset
for idx, entry in enumerate(dataset):
    if "prompt" in entry:
        print(f"Processing prompt {idx + 1}/{len(dataset)}: {entry['prompt']}")
        metadata = fetch_metadata(entry["prompt"])
        
        # Update the entry with the structured metadata if the API call was successful
        if metadata:
            print(f"Updating prompt {idx + 1} with metadata.")
            entry["tags"] = metadata.tags
            entry["negative_tags"] = metadata.negative_tags
            entry["lighting"] = metadata.lighting
            entry["mood"] = metadata.mood
        else:
            print(f"Skipping prompt {idx + 1} due to error.")

        # Save progress after each prompt
        with open("updated_dataset.json", "w") as file:
            json.dump(dataset, file, indent=4)

        # Optional delay to avoid hitting API rate limits
        time.sleep(1)  # Adjust delay based on your rate limits

print("Dataset processing complete. Final file saved to updated_dataset.json")
