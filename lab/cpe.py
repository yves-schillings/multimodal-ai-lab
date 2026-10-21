"""Optional operator-owned activation gate for bounded synthetic local CPEs."""
import json
import os
import math
import re
import time
from pathlib import Path


class CpeInactive(RuntimeError):
    pass


class CpeGate:
    def __init__(self, path):
        self.path = Path(path)

    def state(self):

        # --- READ THE OPERATOR'S APPROVAL, NOT A USER-SUPPLIED FLAG ---
        # The controller mounts this control file read-only. Re-read it so expiry
        # and revocation can take effect without restarting the application.
        # Its schema, phase, expiry, model pins and evidence reference must agree.
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data,dict):
                raise ValueError('Operator control must be an object')
            if (data.get('schema') != 1 or data.get('phase') not in {'provisioned', 'verifying', 'active', 'revoked'}
                    or not isinstance(data.get('approval'), str) or not data['approval']
                    or not isinstance(data.get('expires'), (int, float)) or isinstance(data['expires'], bool)
                    or not math.isfinite(data['expires'])
                    or not self.valid_models(data.get('models')) or not isinstance(data.get('profile_digest'), str)
                    or not re.fullmatch(r'[a-f0-9]{64}',data['profile_digest'])):
                raise ValueError('Invalid operator control')
            if data['expires'] <= time.time():
                return {'phase': 'expired', 'active': False, 'models': []}
            if data['phase'] == 'active' and (not isinstance(data.get('evidence_sha256'), str)
                                             or not re.fullmatch(r'[a-f0-9]{64}',data['evidence_sha256'])):
                raise ValueError('Activation requires accepted evidence')
            return {**data, 'active': data['phase'] == 'active',
                    'scope': 'Synthetic document CPE with approved offline RAG.' if data['models'] else 'Synthetic document/lexical CPE; no model endpoints approved. Storage quota is admission only.'}

        # Missing, unreadable or malformed control data must never enable access.
        # This is the fail-closed default: deny use when approval cannot be checked.
        except (OSError, ValueError, TypeError):
            return {'phase': 'unavailable', 'active': False, 'models': []}

    def authorize(self, actor, path='/api/cases'):

        # --- PROJECT ACTIVATION IS ONE CHECK, NOT ALL ACCESS RIGHTS ---
        # An inactive project cannot serve normal work. Only the reviewer may use
        # the verification phase. The application still checks who may perform
        # which action on each case/document after this project gate succeeds.
        state = self.state()
        if state['phase'] != 'active' and not (state['phase'] == 'verifying' and actor == 'reviewer'):
            raise CpeInactive('CPE is inactive, expired or awaiting independent verification.')
        if path.startswith('/api/models'):
            raise PermissionError('This CPE profile approves no model endpoints or model release actions.')
        return state

    @staticmethod
    def valid_models(models):

        # --- ACCEPT ONLY THE BOUNDED PROFILE IMPLEMENTED BY THIS LAB ---
        # An empty list belongs to the lexical baseline. The AI profile requires
        # these two exact model digests, the pgvector backend and the scoped local
        # gateway address. A model label alone cannot establish artifact identity.
        if models == []:
            return os.environ.get('LAB_RETRIEVAL_BACKEND','lexical') == 'lexical'
        expected = [
            {'name':'bge-m3:latest','digest':'7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab'},
            {'name':'qwen3:4b','digest':'359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'}]
        if models != expected:
            return False
        return (os.environ.get('LAB_RETRIEVAL_BACKEND') == 'pgvector'
                and os.environ.get('LAB_INFERENCE_URL') == 'http://lab-inference-gateway.multimodal-ai-lab.svc:8771'
                and all(os.environ.get(k) == v for k,v in {
                    'LAB_EMBEDDING_MODEL':expected[0]['name'],'LAB_EMBEDDING_DIGEST':expected[0]['digest'],
                    'LAB_GENERATION_MODEL':expected[1]['name'],'LAB_GENERATION_DIGEST':expected[1]['digest']}.items()))


def configured_cpe():

    # Enable this optional gate only for deployments with an operator control
    # file configured. Other deployment modes retain their separate access checks.
    path = os.environ.get('LAB_CPE_STATE_FILE')
    return CpeGate(path) if path else None
