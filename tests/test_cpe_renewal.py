"""Renewal must preserve scope, invalidate old evidence, and fail closed on conflicts."""
import copy
from types import SimpleNamespace
import pytest
from scripts.local_cpe import Controller

class LocalController(Controller):
    def __init__(self):
        self.profile={'key':'a','namespace':'lab-cpe-a','image':'local@sha256:'+'a'*64,
                      'approval':'old-owner-approval','expires':0,'models':[{'name':'approved'}]}
        self.events=[]
        self.conflict=False
    def get(self,kind,name,ns=None):
        if kind=='namespace':return {'metadata':{'labels':{'lab.multimodal.local/cpe':'true'}}}
        return {'metadata':{'resourceVersion':'17'},'data':{'profile.json':__import__('json').dumps(self.profile)}}
    def snapshot(self,key):return {'pvc_uid':'retained','credential_digest':'retained'}
    def save(self,name,data):self.events.append(('save',name,copy.deepcopy(data)))
    def gate(self,key,phase,evidence=None):self.events.append(('gate',phase))
    def oc(self,*args,**kwargs):
        self.events.append(('replace',kwargs['payload']))
        if self.conflict:raise RuntimeError('Concurrent profile change')
        return SimpleNamespace(returncode=0)

@pytest.mark.parametrize('approval,hours',[('',24),('old-owner-approval',24),('new',0),('new',25),('new',True)])
def test_invalid_renewal_cannot_change_the_cluster(approval,hours):
    c=LocalController()
    with pytest.raises(ValueError):c.renew('a',approval,hours)
    assert not c.events

def test_renewal_retains_scope_and_requires_fresh_activation(monkeypatch):
    monkeypatch.setattr('scripts.local_cpe.time.time',lambda:100)
    c=LocalController();c.renew('a','new-owner-approval',2)
    payload=next(x[1] for x in c.events if x[0]=='replace')
    new=__import__('json').loads(payload['data']['profile.json'])
    assert new['expires']==7300 and new['approval']=='new-owner-approval'
    assert {k:v for k,v in new.items() if k not in ('approval','expires')}=={k:v for k,v in c.profile.items() if k not in ('approval','expires')}
    assert payload['metadata']['resourceVersion']=='17'
    assert [x[1] for x in c.events if x[0]=='gate']==['revoked','provisioned']

def test_concurrent_edit_leaves_project_revoked():
    c=LocalController();c.conflict=True
    with pytest.raises(RuntimeError):c.renew('a','new-owner-approval')
    assert [x[1] for x in c.events if x[0]=='gate']==['revoked']
