"""Build the local OKD app with only runtime sources, never local data, models, secrets or venv."""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="lab-app-source-") as directory:
        stage=Path(directory)
        for name in ('Dockerfile','requirements.txt','app.py'):
            shutil.copy2(ROOT/name,stage/name)
        for name in ('lab','static'):
            shutil.copytree(ROOT/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        result=subprocess.run(['oc','start-build','lab-app','-n','multimodal-ai-lab','--from-dir='+str(stage),'--follow'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1200)
        if result.returncode:
            raise RuntimeError('Local app source build failed: '+result.stderr[-1200:])
        print('Local runtime source image built successfully; private data excluded.')


if __name__=='__main__':main()
