from dataclasses import replace
from lab.deployment_policy import ResourcePool, route_decision, deployment_plan


def test_cloud_burst_requires_independent_approval():
    pool = ResourcePool('Cloud Souverain', approved_regions=('approved-region',), available=True)
    assert not route_decision(pool, 'synthetic', 'approved-region', 100)['allowed']
    pool = replace(pool, approved=True, approval_reference='synthetic-demo-approval')
    assert route_decision(pool, 'synthetic', 'approved-region', 100)['allowed']
    assert not route_decision(pool, 'restricted', 'approved-region', 100)['allowed']
    assert not route_decision(pool, 'synthetic', 'other-region', 100)['allowed']
    assert not route_decision(pool, 'synthetic', 'approved-region', 5000)['allowed']
    assert not route_decision(pool, 'synthetic', 'approved-region', 100, requires_cpe=True)['allowed']
    assert not route_decision(replace(pool, available=False), 'synthetic', 'approved-region', 100)['allowed']


def test_cpe_is_case_specific_and_remote_connections_are_disabled():
    pool = ResourcePool('Primary AI Lab', True, 'local-policy', ('restricted',), ('approved-region',), True, available=True)
    assert route_decision(pool, 'restricted', 'approved-region', 100, requires_cpe=True)['allowed']
    assert deployment_plan()['external_connections_enabled'] is False
