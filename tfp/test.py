import matplotlib.pyplot as plt
from menpo.feature import hog, ndfeature # you can import no_op and dsift as well (hog is best for imfrared images though)
IMAGE_PATH = "C:/Users/Mateusz/AppData/Local/Programs/Python/Python36/Scripts/inz/FaceDB_Snapshot_complete" # "/home/temp/schock/Infrared/Databases/FaceDB_Snapshot"
LANDMARK_GROUP = "LJSON" #or "PTS"
features = hog
scales = 1
diagonal = 100

# convert feature from 64 to 32 bit; has no impact on fitting precision but saves 50% memory
# thanks to the menpo team for the hint
# you should define the same for your other features if you use other features than hog in your code
@ndfeature
def float32_hog(x):
    return hog(x).astype(np.float32)

features = float32_hog

from menpo import io as mio
from tqdm import tqdm

print("Importing images")
train_images = []

for i in tqdm(mio.import_images(IMAGE_PATH)):

    # Crop images to Landmarks --> only Face on resulting image
    i = i.crop_to_landmarks_proportion(0.1)

    # Convert multichannel images to greyscale
    if i.n_channels > 2:
        i = i.as_greyscale()

    train_images.append(i)

print("Succesfully imported %d Images" % len(train_images))
plt.imshow(train_images[1])
