import requests
import os
import zipfile
import shutil

output_dir = "data/downloaded/"
os.makedirs(output_dir, exist_ok=True)

URL = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv"
r = requests.get(URL) # create HTTP response object
with open(output_dir+"ChurnData.csv",'wb') as f:
    f.write(r.content)