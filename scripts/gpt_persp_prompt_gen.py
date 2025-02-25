from pydantic import BaseModel
from openai import OpenAI
import base64
import json
import time

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class PromptGen(BaseModel):
    prompt: str

# Initialize OpenAI client
client = OpenAI(api_key="API_KEY")

def get_prompt(pano_prompt, image_path, retries=3, delay=2):
    base64_image = encode_image(image_path)

    for attempt in range(retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Give a concise description of the main elements visible in this part of the image, focusing only on what is physically present. Avoid adding context or interpreting the scene beyond what is shown."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                response_format=PromptGen,
            )
            content = json.loads(response.choices[0].message.content)
            return content["prompt"]
        except Exception as e:
            print(f"Error processing image. Attempt {attempt + 1} of {retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

# Sequential processing function
def process_images_sequentially(dataset, output_file="updated_val.json"):
    updated_dataset = []

    try:
        # Load existing data if the output file exists
        try:
            with open(output_file, "r") as f:
                updated_dataset = json.load(f)
        except FileNotFoundError:
            pass

        for entry_idx, entry in enumerate(dataset):
            print(f"Processing entry {entry_idx+1}/{len(dataset)} - {entry['pano_prompt']}")

            pano_prompt = entry["pano_prompt"]
            images = entry["images"]
            prompts = []

            for idx, image_path in enumerate(images):
                print(f"Processing entry {entry_idx+1}/{len(dataset)} - image {idx+1}/{len(images)}")
                try:
                    prompt = get_prompt(pano_prompt, image_path)
                    print(prompt)
                    prompts.append(prompt)
                except Exception as e:
                    print(f"Failed to process image {idx+1} in entry {entry_idx+1}: {e}")

            # Add updated entry with prompts
            updated_entry = entry.copy()
            updated_entry["prompts"] = prompts
            updated_dataset.append(updated_entry)

            # Write updated dataset after processing each entry
            with open(output_file, "w") as f:
                json.dump(updated_dataset, f, indent=2)

    except Exception as e:
        print(f"Error during processing: {e}")

# Load dataset
with open("val.json", "r") as f:
    dataset = json.load(f)

# Process dataset and write to a new file
process_images_sequentially(dataset)
