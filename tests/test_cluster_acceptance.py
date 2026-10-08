"""Cluster acceptance must not confuse broken probe execution with denied traffic."""
from scripts.openshift_local_acceptance import probe_verdict


def test_missing_pod_cannot_pass_a_denied_path():
    assert probe_verdict(127, 'no running pod for run=lab-probe', False) == ('error', False)


def test_failed_exec_cannot_pass_even_with_a_closed_marker():
    assert probe_verdict(1, 'CLOSED', False) == ('closed', False)


def test_dns_or_command_errors_cannot_establish_network_denial():
    assert probe_verdict(1, 'socket.gaierror: Name or service not known', False) == ('error', False)
    assert probe_verdict(126, 'command unavailable', False) == ('error', False)


def test_confirmed_paths_must_match_the_expected_matrix():
    assert probe_verdict(0, 'OPEN\n', True) == ('open', True)
    assert probe_verdict(0, 'CLOSED\n', False) == ('closed', True)
    assert probe_verdict(0, 'OPEN\n', False) == ('open', False)
    assert probe_verdict(0, 'CLOSED\n', True) == ('closed', False)
