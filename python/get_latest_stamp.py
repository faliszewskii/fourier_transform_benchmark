import glob
import os

def get_latest_stamp(directory):
    pattern = os.path.join(directory, "*.json")
    files = glob.glob(pattern)

    latest_file = ""
    if not files:
        print("No files found.")
    else:
        latest_file = max(files)
    return latest_file