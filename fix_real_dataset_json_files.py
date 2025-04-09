import os
import json
import glob

def fix_json_files(directory_path):
    """
    Fix the 'speed' field in all JSON files in the specified directory,
    converting it from string to integer.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
    """
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    files_processed = 0
    files_modified = 0
    
    for file_path in json_files:
        files_processed += 1
        
        # Read the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Track if this file was modified
            file_modified = False
            
            # Get the timestamp keys (exclude 'QoE' and 'timestamp')
            timestamp_keys = [key for key in data.keys() if key not in ['QoE', 'timestamp']]
            
            # Convert 'speed' from string to integer in each timestamp entry
            for timestamp in timestamp_keys:
                if 'speed' in data[timestamp] and isinstance(data[timestamp]['speed'], str):
                    # Convert string to integer
                    try:
                        data[timestamp]['speed'] = int(data[timestamp]['speed'])
                        file_modified = True
                    except ValueError:
                        print(f"Warning: Could not convert speed value '{data[timestamp]['speed']}' to integer in file {file_path}, timestamp {timestamp}")
            
            # If changes were made, save the file
            if file_modified:
                files_modified += 1
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                    
            # Print progress for every 50 files
            if files_processed % 50 == 0:
                print(f"Processed {files_processed}/{len(json_files)} files")
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    print(f"\nComplete! Processed {files_processed} files.")
    print(f"Modified {files_modified} files with speed field corrections.")

if __name__ == "__main__":
    # Set the directory path to your JSON files
    directory_path = "./real_dataset_2"
    
    # Confirm before proceeding
    print(f"This script will convert 'speed' fields from string to integer in all JSON files in {directory_path}")
    confirmation = input("Do you want to proceed? (y/n): ")
    
    if confirmation.lower() == 'y':
        fix_json_files(directory_path)
    else:
        print("Operation cancelled.")
