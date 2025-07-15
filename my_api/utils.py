import urllib.request

# URL to the ImageNet class index file
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

# Local filename to save it as
filename = "imagenet_class_index.json" 

# Download the file
urllib.request.urlretrieve(url, filename)

print(f"✅ Downloaded: {filename}")
