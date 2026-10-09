"""A real isolated CPU training Job that logs to local MLflow; never promotes a model."""
import json
import sys
from pathlib import Path
sys.path.insert(0,'/app')
from lab.backend_store import LabStore
from lab.models import ModelRegistry

store=LabStore(Path('/tmp/training'))
registry=ModelRegistry(store)
registry.registered_name='local-platform-training-exercise'
result=registry.train()
if not result['gate_passed'] or result['mlflow_status']!='recorded_on_server' or not result['artifact_published']:
    raise RuntimeError('Local training acceptance failed')
Path('/results/training.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['version','accuracy','macro_f1','training_samples','test_samples','mlflow_run_id','artifact_sha256','mlflow_status']}),flush=True)
