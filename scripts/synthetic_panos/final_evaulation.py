import json
import os
import cv2
import time
import shutil

# Define paths
accept_json_path = "accept.json"
final_dataset_json_path = "final_dataset.json"
final_rejected_json_path = "final_rejected.json"
generated_images_dir = os.path.join(os.getcwd(), "generated_images")
final_generated_images_dir = os.path.join(os.getcwd(), "final_generated_images")
check_interval = 10  # Interval to recheck if accept.json is empty
batch_size = 5  # Number of images to load at a time to avoid memory issues

# Load JSON data
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

# Save JSON data
def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Function to wrap text based on pixel width
def wrap_text(text, font, font_scale, font_thickness, max_width):
    words = text.split()
    wrapped_lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        text_size, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if text_size[0] <= max_width:
            line = test_line
        else:
            wrapped_lines.append(line)
            line = word

    if line:
        wrapped_lines.append(line)

    return wrapped_lines

# Function to display image with prompt and wait for 'y' or 'n' input directly on the image window
def display_image_with_prompt(image_path, prompt_text, queue_length):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5  # Adjusted for readability
    font_thickness = 4
    padding = 10
    line_spacing = 5
    max_width = img.shape[1] - 2 * padding

    full_text = f"Prompt: {prompt_text} | Remaining: {queue_length}"
    wrapped_text = wrap_text(full_text, font, font_scale, font_thickness, max_width)

    text_height = int((cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + line_spacing) * len(wrapped_text))
    overlay_height = text_height + padding * 3

    img_with_text = cv2.copyMakeBorder(img, overlay_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    y_pos = padding + text_height // len(wrapped_text)
    for line in wrapped_text:
        cv2.putText(img_with_text, line, (padding, y_pos), font, font_scale, (255, 255, 255), font_thickness)
        y_pos += text_height // len(wrapped_text) + line_spacing

    cv2.imshow("Image Review", img_with_text)

    while True:
        key = cv2.waitKey(0)
        if key == ord('y'):
            cv2.destroyAllWindows()
            return 'y'
        elif key == ord('n'):
            cv2.destroyAllWindows()
            return 'n'

# Main processing loop
while True:
    accept_entries = load_json(accept_json_path)

    if not accept_entries:
        print("No entries in accept.json. Rechecking in 10 seconds...")
        time.sleep(check_interval)
        continue

    # Process images in batches
    while accept_entries:
        batch = accept_entries[:batch_size]  # Take the next batch
        for entry in batch:
            image_path = os.path.join(generated_images_dir, entry["image"].lstrip("/"))
            prompt_text = entry["prompt"]
            queue_length = len(accept_entries)

            print(f"Displaying image: {image_path}")

            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found. Skipping...")
                accept_entries.remove(entry)
                save_json(accept_entries, accept_json_path)
                continue

            response = display_image_with_prompt(image_path, prompt_text, queue_length)
            if response is None:
                continue

            # Remove the processed entry from accept_entries
            accept_entries = load_json(accept_json_path)
            accept_entries.remove(entry)
            save_json(accept_entries, accept_json_path)

            if response == 'y':
                print(f"Image accepted: {image_path}")
                # Move the image folder to final_generated_images_dir
                # Assuming that the image folder is the immediate parent of the image
                image_folder = os.path.dirname(image_path)
                folder_name = os.path.basename(image_folder)
                dest_folder = os.path.join(final_generated_images_dir, folder_name)

                if not os.path.exists(final_generated_images_dir):
                    os.makedirs(final_generated_images_dir)

                try:
                    shutil.move(image_folder, dest_folder)
                    print(f"Moved folder {image_folder} to {dest_folder}")
                except Exception as e:
                    print(f"Error moving folder {image_folder} to {dest_folder}: {e}")

                # Add entry to final_dataset.json
                final_dataset_entries = load_json(final_dataset_json_path)
                final_dataset_entries.append(entry)
                save_json(final_dataset_entries, final_dataset_json_path)

            elif response == 'n':
                print(f"Image rejected: {image_path}")
                # Add entry to final_rejected.json
                final_rejected_entries = load_json(final_rejected_json_path)
                final_rejected_entries.append(entry)
                save_json(final_rejected_entries, final_rejected_json_path)

            print("Updated accept.json, final_dataset.json, and final_rejected.json.")

        # Reload accept_entries in case of external updates and break if empty
        accept_entries = load_json(accept_json_path)
        if not accept_entries:
            break

    print("Finished processing current entries in accept.json.")
