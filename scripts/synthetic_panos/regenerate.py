import pickle
import os
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure WebDriver (e.g., Chrome)
driver = webdriver.Chrome()  # Ensure chromedriver is in your PATH

# Define paths
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
generated_images_dir = os.path.join(os.getcwd(), "generated_images")
cookie_file_path = "cookies.pkl"  # File to store cookies
reject_json_path = "reject.json"  # File to check for rejected entries
queue_json_path = "queue.json"  # Output file for queue

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Function to save JSON data
def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Create directory for generated images if it doesn't exist
os.makedirs(generated_images_dir, exist_ok=True)

def save_cookies(driver, path):
    with open(path, "wb") as file:
        pickle.dump(driver.get_cookies(), file)

def load_cookies(driver, path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            cookies = pickle.load(file)
            for cookie in cookies:
                driver.add_cookie(cookie)

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.png')]
    return max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f))) if files else None

print("Opening Blockade Labs Skybox website...")
driver.get("https://skybox.blockadelabs.com/starter/eece4e2a09ac445932b4590e5b53871e")

# Load cookies if they exist, otherwise prompt for login
if os.path.exists(cookie_file_path):
    load_cookies(driver, cookie_file_path)
    driver.refresh()  # Refresh to apply cookies
else:
    # If no cookies, prompt user to log in and save cookies afterward
    input("Please log in manually, then press Enter here in the console to continue...")
    save_cookies(driver, cookie_file_path)
    driver.refresh()

try:
    while True:
        # Load reject.json
        reject_entries = load_json(reject_json_path)
        
        # Check if there are any entries to process
        if not reject_entries:
            print("No entries in reject.json. Checking again in 10 seconds...")
            time.sleep(10)
            continue

        # Process each entry in reject.json
        for entry in reject_entries[:]:  # Use a copy of the list for safe iteration
            # Construct the prompt and negative tags
            tags_str = ", ".join(entry["tags"])
            final_prompt = f"{entry['prompt']}, {tags_str}, {entry['lighting']} lighting, {entry['mood']} mood, massive scene, detailed"
            negative_tags_str = ", ".join(entry["negative_tags"])

            # Print which prompt is being processed
            print(f"Processing prompt: {final_prompt}")
            print(f"Negative tags: {negative_tags_str}")

            # Enter prompt and negative tags
            prompt_input = WebDriverWait(driver, 30).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[placeholder='Dream up your world']"))
            )
            promptX = driver.find_element(By.CSS_SELECTOR, "svg.absolute.right-1.top-1\\/2.-translate-y-1\\/2.transform.cursor-pointer")
            promptX.click()
            prompt_input.send_keys(final_prompt)

            # Enter negative tags
            negative_input = WebDriverWait(driver, 30).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "input[placeholder='Describe what to avoid']"))
            )
            Xelems = driver.find_elements(By.CSS_SELECTOR, "svg.absolute.right-1.top-1\\/2.-translate-y-1\\/2.transform.cursor-pointer")
            if len(Xelems) > 1:
                negX = Xelems[1]
                negX.click()
            negative_input.send_keys(negative_tags_str)

            # Submit the prompt
            prompt_input.send_keys(Keys.RETURN)

            # Wait for the download button
            print("Waiting for the download button...")
            download_button = WebDriverWait(driver, 120).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Download']]"))
            )
            time.sleep(1)
            download_button.click()

            # Wait for the secondary download button
            print("Waiting for the secondary download button...")
            secondary_download_button = WebDriverWait(driver, 120).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='PNG']]"))
            )
            time.sleep(1)
            secondary_download_button.click()
            time.sleep(1)

            # Check and replace the downloaded image
            latest_file = get_latest_file(download_dir)
            while not latest_file or latest_file == "PLACEHOLDER.txt":
                latest_file = get_latest_file(download_dir)
                time.sleep(1)

            if latest_file:
                folder_name = entry['prompt'].replace(" ", "_")
                new_folder = os.path.join(generated_images_dir, folder_name)
                os.makedirs(new_folder, exist_ok=True)

                local_image_path = os.path.join(new_folder, "image.png")
                os.replace(os.path.join(download_dir, latest_file), local_image_path)
                entry["image"] = f"/{folder_name}/image.png"
                
                print(f"Image regenerated and saved to {local_image_path}")

                # Add the entry back to queue.json
                queue_entries = load_json(queue_json_path)  # Reload to ensure latest state
                queue_entries.append(entry)
                save_json(queue_entries, queue_json_path)
                print(f"Entry for '{entry['prompt']}' added to the end of queue.json")

                # Remove the processed entry from reject.json and save immediately
                reject_entries = load_json(reject_json_path)
                reject_entries.remove(entry)
                save_json(reject_entries, reject_json_path)
                print(f"Entry for '{entry['prompt']}' removed from reject.json")

            else:
                print("Error: Could not find downloaded file.")
            
            # Close the download modal
            Xbutton = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Close']]"))
            )
            Xbutton.click()
            print("Close button clicked successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Save cookies and close the driver
    save_cookies(driver, cookie_file_path)
    driver.quit()
    print("WebDriver closed.")
