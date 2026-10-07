"""Container checks run as an arbitrary OpenShift-style UID, without network access.

Run inside the application image:
docker run --rm --network none --user 1000780000:0 --cap-drop ALL
  --security-opt no-new-privileges -e LAB_DATA_DIR=/app/data
  --entrypoint python IMAGE /acceptance/container_acceptance.py
Mount this script at /acceptance/container_acceptance.py read-only.
This checks image portability, not an OpenShift AI cluster installation.
"""
import os
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from PIL import Image, ImageDraw, ImageFont
from app import create_app
from lab.documents import extract_document


def main():
    assert os.getuid() != 0, 'Must run as non-root.'
    app = create_app(Path(os.environ['LAB_DATA_DIR']))
    assert app.state.store.root.is_dir()
    assert app.state.registry.state()['available']
    with tempfile.TemporaryDirectory() as temp:
        image = Image.new('RGB', (1200, 220), 'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
        draw.text((35, 50), 'Synthetic evidence inventory: blue bicycle.', fill='black', font=font)
        source = Path(temp) / 'synthetic-scan.png'
        image.save(source)
        extracted = extract_document(source)
        assert 'blue bicycle' in extracted['text'].lower(), extracted
    print('PASS arbitrary non-root UID, writable data store, classifier imports and synthetic image OCR')


if __name__ == '__main__':
    main()
