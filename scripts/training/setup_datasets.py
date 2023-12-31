import os
import urllib.request
import shutil
import zipfile
import argparse


parser = argparse.ArgumentParser(description='.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
parser.add_argument('--foo', action='store_true')
args = parser.parse_args()


# URL of the ZIP file.
data_url = 'https://github.com/giacomo-janson/idpsam/releases/download/example1.0.0/sam_training_example.zip'
# Output file path.
zip_fp = os.path.join('sam_training_example.zip')

# Open a connection to the URL.
with urllib.request.urlopen(data_url) as response:
    print(f"- Downloading data from: {data_url}")
    # Ensure the request was successful.
    if response.status == 200:
        print("- Saving data")
        with open(zip_fp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    else:
        raise OSError(f"- Failed to download: Status code {response.status}")

# Directory where you want to extract the contents
extract_dp = 'ok/'

# Ensure the extraction directory exists
os.makedirs(extract_dp, exist_ok=True)

# Open the zip file.
with zipfile.ZipFile(zip_fp, 'r') as zip_ref:
    # Extract contents.
    zip_ref.extractall(extract_dp)
print(f"- Contents extracted to {extract_dp}")

os.remove(zip_fp)
print("- Done.")