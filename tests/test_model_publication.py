"""Release acceptance: actual MLflow artifact round-trip and fail-closed gates."""
import json
from unittest.mock import patch

import mlflow.sklearn
import pytest

from app import create_app


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    return create_app(tmp_path).state.registry


def test_published_model_loads_and_matches_inference(registry, tmp_path):
    candidate = registry.train()
    assert candidate['artifact_published'], candidate
    client = registry._client(candidate)
    artifact = client.download_artifacts(candidate['mlflow_run_id'], 'model', str(tmp_path))
    reloaded = mlflow.sklearn.load_model(artifact)
    assert reloaded.predict(['Dear colleague please reply to this email'])[0] == 'correspondence'
    registry.promote('reviewer', candidate['version'])
    assert registry.classify('Dear colleague please reply to this email')['label'] == 'correspondence'
    registered = client.get_model_version(candidate['mlflow_model_name'], candidate['mlflow_model_version'])
    assert registered.tags['local_version'] == candidate['version']


def test_failed_artifact_publication_blocks_promotion(registry):
    with patch('mlflow.tracking.MlflowClient.log_artifacts', side_effect=OSError('unavailable')):
        candidate = registry.train()
    assert not candidate['artifact_published']
    assert candidate['gate_passed']
    with pytest.raises(ValueError, match='publication'):
        registry.promote('reviewer', candidate['version'])
    assert registry.state()['active_version'] is None


def test_modified_local_artifact_blocks_promotion(registry):
    candidate = registry.train()
    (registry.root / (candidate['version'] + '.pkl')).write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum'):
        registry.promote('reviewer', candidate['version'])
    assert registry.state()['active_version'] is None


def test_missing_remote_artifact_preserves_active_version(registry):
    first, second = registry.train(), registry.train()
    registry.promote('reviewer', first['version'])
    with patch('mlflow.tracking.MlflowClient.download_artifacts', side_effect=OSError('unavailable')):
        with pytest.raises(ValueError, match='blocked'):
            registry.promote('reviewer', second['version'])
    assert registry.state()['active_version'] == first['version']
    registry.promote('reviewer', second['version'])
    with patch('mlflow.tracking.MlflowClient.get_model_version', side_effect=OSError('unavailable')):
        with pytest.raises(ValueError, match='blocked'):
            registry.rollback('reviewer')
    assert registry.state()['active_version'] == second['version']


def test_legacy_unpublished_candidate_cannot_be_released(registry):
    candidate = registry.train()
    candidate.pop('artifact_published')
    with registry.store._connection(write=True) as db:
        db.execute('UPDATE model_versions SET metadata_json=? WHERE version=?',
                   (json.dumps(candidate), candidate['version']))
    with pytest.raises(ValueError, match='publication'):
        registry.promote('reviewer', candidate['version'])
