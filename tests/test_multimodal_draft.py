import pytest
from lab.backend_store import LabStore


def test_document_only_draft_and_separation(tmp_path):
    store=LabStore(tmp_path)
    case=store.create_case('officer-a','Synthetic document-only case')
    result=store.add_document('officer-a',case['id'],'Inventory',{'text':'Synthetic blue bicycle photograph.','engine':'utf8'},{})
    with pytest.raises(ValueError):
        store.make_draft('officer-a',case['id'])
    case=store.get_case('reviewer',case['id'])
    case=store.review_document('reviewer',case['id'],result['document_id'],None,case['revision'])
    case=store.make_draft('reviewer',case['id'])
    with pytest.raises(PermissionError):
        store.approve('reviewer',case['id'],case['revision'])
    case=store.review_document('officer-a',case['id'],result['document_id'],None,case['revision'])
    case=store.make_draft('officer-a',case['id'])
    assert case['draft']['sources'][0]['id']==result['document_id']
    approved=store.approve('reviewer',case['id'],case['revision'])
    assert approved['status']=='approved'
    changed=store.review_document('officer-a',case['id'],result['document_id'],'Synthetic updated source.',approved['revision'])
    assert changed['approved_revision'] is None and changed['draft'] is None


def test_unreviewed_document_blocks_transcript_draft(tmp_path):
    store=LabStore(tmp_path)
    case=store.get_case('officer-a','demo-case-a')
    for source in case['segments']:
        case=store.update_segment('officer-a',case['id'],source['id'],source['text'],case['revision'])
    store.add_document('officer-a',case['id'],'Unreviewed source',{'text':'Fictional correspondence.','engine':'utf8'},{})
    with pytest.raises(ValueError):
        store.make_draft('officer-a',case['id'])
