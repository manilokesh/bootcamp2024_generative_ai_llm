
# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel

llm = init_llm(LlmModel.MISTRAL)

# endregion Initialise foundation LLM

########################################################################


import os
from yt_dlp import YoutubeDL
import whisper

url = "https://www.youtube.com/watch?v=Rb9Bpw8yvTg"
save_dir = "data/youtube/"
audio_file = os.path.join(save_dir, "audio4")

os.makedirs(save_dir, exist_ok=True)


def download_audio(url, save_path, audio_extension):

    ydl_opts = {
        "final_ext": audio_extension,
        "format": "bestaudio/best",
        "outtmpl": save_path + "/"+"%(title)s-%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "nopostoverwrites": False,
                "preferredcodec": audio_extension,
                "preferredquality": "320",
            },
            {
                "add_chapters": True,
                "add_infojson": "if_exists",
                "add_metadata": True,
                "key": "FFmpegMetadata",
            },
            {"already_have_thumbnail": False, "key": "EmbedThumbnail"},
        ],
        "ffmpeg_location": os.path.realpath("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])        
        info = ydl.extract_info(url, download=True) 
        file_path = ydl.prepare_filename(info) 
        return file_path

audio_extension = 'mp3'
downloaded_file_name = download_audio(url, audio_file,audio_extension)

def replace_file_extension(file_path, new_extension):
    # Split the file path into root and extension
    root, _ = os.path.splitext(file_path)
    
    # Ensure the new extension starts with a dot
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    
    # Combine the root with the new extension
    new_file_path = root + new_extension
    
    return new_file_path

new_extension = 'wav'
downloaded_file_name = replace_file_extension(downloaded_file_name, audio_extension)

print(f"DOWNLOADED FILE NAME: {downloaded_file_name}")


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel


def transcribe_audio(audio_path):
    model = WhisperModel("medium")
    segments, info = model.transcribe(audio_path)
    return segments, info


segments, info = transcribe_audio(downloaded_file_name)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))



"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from faster_whisper import WhisperModel

model = WhisperModel("medium")

segments, info = model.transcribe("data/youtube/LLM Apps： Overcoming the Context Window limits.m4a")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
"""


"""
for windows

in powershell
> Set-ExecutionPolicy AllSigned or Set-ExecutionPolicy Bypass -Scope Process.

Now run the following command:

> Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

install 
> choco install ffmpeg

installed in 
C:\ProgramData\chocolatey\bin

add the below to ydl_opts
'ffmpeg_location':os.path.realpath('C:/ProgramData/chocolatey/bin/ffmpeg.exe'), 

"""
