import json
import os
import cv2
import time

# Define paths
queue_json_path = "queue.json"
accept_json_path = "accept.json"
reject_json_path = "reject.json"
generated_images_dir = os.path.join(os.getcwd(), "generated_images")
check_interval = 10  # Interval to recheck if queue is empty
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
    font_scale = 5
    font_thickness = 2
    padding = 10
    line_spacing = 5
    max_width = img.shape[1] - 2 * padding

    full_text = f"Prompt: {prompt_text} | Queue Remaining: {queue_length}"
    wrapped_text = wrap_text(full_text, font, font_scale, font_thickness, max_width)

    text_height = int((cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + line_spacing) * len(wrapped_text))
    overlay_height = text_height + padding * 3 

    img_with_text = cv2.copyMakeBorder(img, overlay_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    y_pos = padding + text_height // len(wrapped_text)
    for line in wrapped_text:
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            cv2.putText(img_with_text, line, (padding + dx, y_pos + dy), font, font_scale, (255, 255, 255), font_thickness)
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
    queue_entries = load_json(queue_json_path)

    if not queue_entries:
        print("No entries in queue.json. Rechecking in 10 seconds...")
        time.sleep(check_interval)
        continue

    # Process images in batches
    while queue_entries:
        batch = queue_entries[:batch_size]  # Take the next batch
        for entry in batch:
            rejected = False
            image_path = os.path.join(generated_images_dir, entry["image"].lstrip("/"))
            prompt_text = entry["prompt"]
            queue_length = len(queue_entries)

            print(f"Displaying image: {image_path}")

            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found. Skipping...")
                continue

            response = display_image_with_prompt(image_path, prompt_text, queue_length)
            if response is None:
                continue

            if response == 'y':
                print(f"Image accepted: {image_path}")
            elif response == 'n':
                print(f"Image rejected: {image_path}")
                rejected = True

                
            queue_entries = load_json(queue_json_path)
            queue_entries.remove(entry)  # Remove processed entry

            if rejected:
                reject_entries = load_json(reject_json_path)
                reject_entries.append(entry)
                save_json(reject_entries, reject_json_path)
            else:
                accept_entries = load_json(accept_json_path)
                accept_entries.append(entry)
                save_json(accept_entries, accept_json_path)

            save_json(queue_entries, queue_json_path)

            print("Queue, accept, and reject files updated.")

        # Reload JSON files in case of external updates and break if queue is empty
        queue_entries = load_json(queue_json_path)
        if not queue_entries:
            break

    print("Finished processing current entries in queue.json.")
