# Step 1: Install Necessary Libraries
#!pip install torch diffusers accelerate

# Step 2: Load the Pre-trained Model
import torch
from diffusers import DiffusionPipeline

#pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
#pipe = pipe.to("cuda")
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",  variant="fp16")

# Step 3: Generate a Video
prompt = "red Penguine dancing happily"
# Generate more frames by running the pipeline multiple times
num_iterations = 4  # Number of times to run the pipeline for more frames
all_frames = []

for _ in range(num_iterations):
    video_frames = pipe(prompt).frames[0]
    all_frames.extend(video_frames)

# Step 4: Export the Video
from diffusers.utils import export_to_video

video_path = export_to_video(all_frames)
print(f"Video saved at: {video_path}")

# Step 5: Download the Video (Optional for Google Colab)
from google.colab import files

files.download(video_path)