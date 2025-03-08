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
output_json_path = "updated_dataset.json"  # Output JSON file with local image paths


# Load dataset
with open("cleaned_dataset.json") as f:
    dataset = json.load(f)

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
    # Filter for files that end with .png or are named PLACEHOLDER.txt
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and
        (f.endswith('.png') or f == 'PLACEHOLDER.txt')
    ]
    # If there are no matching files, return None
    if not files:
        return None
    # Get the latest file based on modification time
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return latest_file

previous_latest_file = "PLACEHOLDER.txt"

try:
    # Step 1: Open the website
    print("Opening Blockade Labs Skybox website...")
    driver.get("https://skybox.blockadelabs.com/starter/eece4e2a09ac445932b4590e5b53871e")
    
    # Step 2: Load cookies if they exist
    if os.path.exists(cookie_file_path):
        load_cookies(driver, cookie_file_path)
        driver.refresh()  # Refresh to apply cookies

    # Check if login is required
    try:
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[placeholder='Dream up your world']"))
        )
    except:
        input("Please log in manually, then press Enter here in the console to continue...")
        save_cookies(driver, cookie_file_path)
        driver.refresh()

    # Iterate through each entry in the dataset
    for entry in dataset:
        # Construct the prompt and negative tags
        tags_str = ", ".join(entry["tags"])
        final_prompt = f"{entry['prompt']}, {tags_str}, {entry['lighting']} lighting, {entry['mood']} mood"
        negative_tags_str = ", ".join(entry["negative_tags"])
        
        # Print which prompt is being processed
        print(f"Processing prompt: {final_prompt}")
        print(f"Negative tags: {negative_tags_str}")

        # Step 3: Enter prompt and negative tags
        prompt_input = WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[placeholder='Dream up your world']"))
        )
        
        # Clear the prompt field and enter the new prompt
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

        # Step 4: Wait for the download button to become clickable
        print("Waiting for the download button...")
        download_button = WebDriverWait(driver, 120).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Download']]"))
        )
        time.sleep(1)
        download_button.click()

        print("Download button clicked successfully.")

        # Step 5: Wait for the secondary download button to become clickable and click it
        print("Waiting for the secondary download button to become available...")
        secondary_download_button = WebDriverWait(driver, 120).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='PNG']]"))
        )
        time.sleep(1)
        secondary_download_button.click()
        time.sleep(1)
        
        latest_file = get_latest_file(download_dir)

        print(f"Latest file: {latest_file}")
        print(f"Previous latest file: {previous_latest_file}")
        while latest_file == previous_latest_file:
            latest_file = get_latest_file(download_dir)
            print(f"Latest file: {latest_file}")
            print(f"Previous latest file: {previous_latest_file}")
            time.sleep(1)
            
        
        time.sleep(1)
        if latest_file:
            folder_name = entry['prompt'].replace(" ", "_")
            new_folder = os.path.join(generated_images_dir, folder_name)
            os.makedirs(new_folder, exist_ok=True)
            
            # Define the path to save the image
            path_name = "/" + folder_name + "/image.png"
            local_image_path = os.path.join(new_folder, "image.png")
            os.rename(os.path.join(download_dir, latest_file), local_image_path)
            
            # Add the local path to the entry in the dataset
            entry["image"] = path_name
            
            print(f"Image downloaded and saved to {local_image_path}")
        else:
            print("Error: Could not find downloaded file.")
        
        with open(output_json_path, "r") as f:
            all_entries = json.load(f)

        all_entries.append(entry)

        with open(output_json_path, "w") as f:
            json.dump(all_entries, f, indent=4)

        print(f"Updated dataset saved to {output_json_path}")
        # Close the download modal
        Xbutton = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Close']]"))
        )
        Xbutton.click()
        print("Close button clicked successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Save the updated dataset with local image paths to a new JSON file
    driver.quit()
