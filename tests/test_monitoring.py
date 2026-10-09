"""Canary detects regression, refuses altered artifacts and never silently passes MLflow failures."""
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lab.backend_store import LabStore
from lab.models import TRAIN, TEST
from lab.monitoring import CANARY, QualityMonitor


def configured(tmp_path,model):
    root=tmp_path/'models';root.mkdir()
    blob=pickle.dumps(model);(root/'classifier-test.pkl').write_bytes(blob)
    candidate={'version':'classifier-test','artifact_sha256':hashlib.sha256(blob).hexdigest()}
    client=Mock();client.get_experiment_by_name.return_value=SimpleNamespace(experiment_id='1')
    client.create_run.return_value=SimpleNamespace(info=SimpleNamespace(run_id='canary-run'))
    registry=Mock(root=root,store=LabStore(tmp_path/'data'))
    registry.state.return_value={'active_version':'classifier-test','versions':[candidate]}
    registry._client.return_value=client
    return QualityMonitor(registry),registry,client


def test_fresh_canary_does_not_reuse_train_or_release_holdout():
    canary={text for values in CANARY.values() for text in values}
    assert not canary.intersection(text for values in TRAIN.values() for text in values)
    assert not canary.intersection(text for values in TEST.values() for text in values)


def test_real_classifier_scored_and_only_metrics_logged(tmp_path):
    x,y=[],[]
    for label,values in TRAIN.items():x.extend(values);y.extend([label]*len(values))
    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('classifier',LogisticRegression(C=8,random_state=17))]).fit(x,y)
    monitor,registry,client=configured(tmp_path,model)
    result=monitor.run()
    assert result['status']=='healthy' and result['metrics']['sample_count']==8
    assert result['mlflow_run_id']=='canary-run' and len(monitor.state()['runs'])==1
    logged=str(client.mock_calls)
    assert all(text not in logged for values in CANARY.values() for text in values)
    registry._write_state.assert_not_called()


def test_degraded_classifier_alerts_without_release_change(tmp_path):
    model=DummyClassifier(strategy='constant',constant='correspondence').fit(['a','b'],['correspondence','incident_report'])
    monitor,registry,_=configured(tmp_path,model)
    result=monitor.run()
    assert result['status']=='alert' and result['metrics']['accuracy']==0.25
    registry._write_state.assert_not_called()


def test_changed_model_is_not_unpickled(tmp_path,monkeypatch):
    monitor,registry,client=configured(tmp_path,{'not':'a model'})
    (registry.root/'classifier-test.pkl').write_bytes(b'changed')
    loader=Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr('lab.monitoring.pickle.loads',loader)
    assert monitor.run()['status']=='unavailable'
    loader.assert_not_called();client.create_run.assert_not_called()


def test_mlflow_failure_is_unavailable_not_healthy(tmp_path):
    model=DummyClassifier(strategy='constant',constant='correspondence').fit(['a','b'],['correspondence','incident_report'])
    monitor,_,client=configured(tmp_path,model)
    client.log_metric.side_effect=RuntimeError('server offline')
    assert monitor.run()['status']=='unavailable'
    client.set_terminated.assert_called_with('canary-run','FAILED')
