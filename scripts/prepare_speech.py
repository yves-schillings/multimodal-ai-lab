"""Explicit public model acquisition; no recordings are sent anywhere."""
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from faster_whisper.utils import download_model
target=Path(__file__).resolve().parents[1]/'models'/'whisper-base'
target.mkdir(parents=True,exist_ok=True)
download_model('base',output_dir=str(target))
print('Local speech model prepared:',target)
