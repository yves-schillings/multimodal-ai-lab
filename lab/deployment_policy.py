"""Fail-closed planning policy. No remote connector is enabled by this module."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourcePool:
    name: str
    approved: bool = False
    approval_reference: str = ''
    allowed_data_classes: tuple[str, ...] = ('synthetic',)
    approved_regions: tuple[str, ...] = ()
    supports_cpe: bool = False
    max_request_tokens: int = 4096
    available: bool = False


def route_decision(pool, data_class, region, token_budget, requires_cpe=False):
    if data_class not in ('synthetic', 'internal', 'restricted'):
        raise ValueError('Unknown data classification')
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError('Token budget must be a positive integer')
    checks = (
        (pool.approved and bool(pool.approval_reference), 'Destination approval is required'),
        (data_class in pool.allowed_data_classes, 'Data classification is not permitted'),
        (region in pool.approved_regions, 'Processing region is not approved'),
        (not requires_cpe or pool.supports_cpe, 'An approved CPE is required'),
        (token_budget <= pool.max_request_tokens, 'Request exceeds its inference token quota'),
        (pool.available, 'Pool unavailable; no automatic external fallback'),
    )
    for passed, reason in checks:
        if not passed:
            return {'allowed': False, 'reason': reason, 'destination': pool.name}
    return {'allowed': True, 'reason': 'Planning policy passed; live connector is not implemented', 'destination': pool.name}


def deployment_plan():
    return {
        'initial_platform': 'Red Hat OpenShift AI',
        'platform_status': 'target_blueprint_not_deployed',
        'resources': ['Primary AI Lab', 'Secondary GPU Data Center', 'Cloud Souverain'],
        'external_connections_enabled': False,
        'service_chain': ['Primary AI Lab', 'Secondary GPU Data Center', 'Cloud Souverain'],
        'direct_primary_to_sovereign_route': False,
        'burst_owner': 'Secondary GPU Data Center; onward processing requires explicit approval',
        'cloud_burst': 'Explicitly approved jobs only; classification and processing geography precede capacity routing',
        'access_tokens': 'Short-lived audience-bound scoped access tokens; secrets remain server-side',
        'consumption': 'Model input/output token quotas; audio duration and GPU time quotas for other workloads',
        'cpe': 'Controlled Project Environment selected by case requirements; not mandatory for every speech job',
        'network': 'Approved private interconnect or VPN, TLS and default-deny egress',
        'transfer': 'Each destination and onward transfer needs its own approval; no implicit cloud fallback',
        'azure': 'Later preparation; real data requires explicit authorisation',
    }
