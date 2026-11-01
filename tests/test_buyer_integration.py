"""Fund-buyer integration seams on isolated synthetic public workspaces."""
from copy import deepcopy
import json

import pytest

from rmbfund.buyer_research import followed_review, followed_state, install_buyer_config, load_buyer_config
from rmbfund.domain import ValidationError, now_iso
from rmbfund.position_sizing import conditional_size
from rmbfund.service import Workspace
from test_investor_decision import HOLDING, SUBJECT, thesis
from rmbfund.personal_assessment import build_assessment


@pytest.mark.parametrize('constraint', ['spendable_cash', 'max_investment', 'exposure_cap'])
def test_explicit_zero_blocks_add_without_inventing_missing_risk_inputs(constraint):
    sizing = conditional_size({constraint: '0'})
    result = build_assessment(None, subject=SUBJECT, holding=HOLDING,
                              research_thesis=thesis('accumulate'), sizing_plan=sizing)
    branches = result['investor_decision']['branches']
    assert branches['flat']['action'] == branches['addition']['action'] == 'wait'
    assert result['forecast'] == {}
    assert sizing['whole_fund']['scenario_return'] is None


def test_config_versions_are_actual_available_and_bad_containers_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr('rmbfund.buyer_research.now_iso', lambda: '2026-10-25T10:00:00+08:00')
    first = {'peer_sets': {}, 'fund_metadata': {'x': {'currency': 'CNY'}}, 'source_plan': []}
    install_buyer_config(tmp_path, first)
    assert load_buyer_config(tmp_path, as_of='2026-10-24T10:00:00+08:00')['fund_metadata'] == {}
    monkeypatch.setattr('rmbfund.buyer_research.now_iso', lambda: '2026-10-26T10:00:00+08:00')
    install_buyer_config(tmp_path, {'peer_sets': {}}, replace=True)
    assert load_buyer_config(tmp_path, as_of='2026-10-25T12:00:00+08:00') == first
    for bad in ({'peer_sets': []}, {'fund_metadata': []}, {'source_plan': {}}, {'peer_sets': {'x': {}}}):
        with pytest.raises(ValidationError):
            install_buyer_config(tmp_path, bad, replace=True)
    assert len(list((tmp_path / 'buyer-config-versions').glob('*.json'))) == 2


def test_disclosure_import_and_change_review_survive_restart_and_other_fund_refresh(tmp_path, monkeypatch):
    from dataclasses import asdict
    clock = now_iso()
    workspace = Workspace(tmp_path / 'synthetic-public', mode='public')
    a = workspace.add_product('SYNTHA', '合成基金A', 'open_fund', share_class='C')
    b = workspace.add_product('SYNTHB', '合成基金B', 'open_fund', share_class='C')
    workspace.store.initialize_evidence_schema()
    jobs = [{'job_id': 'job-' + key, 'product_id': product, 'family': 'disclosure',
             'source_url': 'https://example.org/' + key, 'parser': 'reviewed_extract_v1', 'config': {}}
            for key, product in [('a', a), ('b', b)]]
    hash_a, hash_b = 'a' * 64, 'b' * 64
    state = {'hashes': {'job-a': hash_a, 'job-b': hash_b}, 'blocked_urls': [], 'updates': []}
    log = workspace.root / 'followed-state.json'
    log.write_text(json.dumps(state))
    raw = tmp_path / 'reviewed-public.json'
    raw.write_text(json.dumps({'managers': [{'name': '合成经理', 'start_date': '2024-01-01'}]}))
    import hashlib
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    changed = True

    def fake(subjects, families, planned, budget, output, **kwargs):
        assert len(subjects) == len(planned) == 1
        assert set(kwargs['previous_hashes']) == {planned[0]['job_id']}
        output.mkdir()
        job = planned[0]
        return {'completed_at': clock, 'outcomes': [{**job, 'status': 'ok', 'http_status': 200,
            'raw_path': str(raw), 'raw_sha256': raw_hash, 'changed': changed,
            'observed_at': clock, 'available_at': clock, 'receipt_path': str(output / 'receipt.json'),
            'records': [{'evidence_import': {'family': 'management', 'report_date': '2026-10-25',
                'payload': {'managers': [{'name': '合成经理', 'start_date': '2024-01-01'}]},
                'locator': 'json:/managers'}}]}]}

    monkeypatch.setattr('rmbfund.providers.followed.refresh_followed', fake)
    identity = asdict(workspace.store.get_product(a))
    result = workspace.refresh_followed([identity], families=['disclosure'], source_plan=jobs,
                                        output_dir=tmp_path / 'first')
    assert result['evidence_added'] == 1
    assert workspace.store.advice_evidence(a, now_iso())[0]['family'] == 'management'
    assert followed_state(workspace.root)['hashes']['job-b'] == hash_b
    workspace.store.close()
    workspace = Workspace(workspace.root, mode='public', initialize=False)
    changed = False
    workspace.refresh_followed([identity], families=['disclosure'], source_plan=jobs,
                               output_dir=tmp_path / 'second')
    stored = followed_state(workspace.root)
    assert stored['hashes']['job-b'] == hash_b
    assert stored['updates'][-1]['unreviewed_change_at'] == clock
    review = thesis('maintain')
    review['source'] = {'available_at': '2026-10-24T10:00:00+08:00'}
    projected = followed_review(review, stored, product_id=a, as_of=now_iso())
    assert any('未复核变化' in text for text in projected['needed_review'])
    old = deepcopy(projected)
    reviewed = deepcopy(review)
    reviewed['payload']['citations'] = [{'publisher_url': jobs[0]['source_url'], 'checked_at': now_iso()}]
    reviewed['source']['available_at'] = now_iso()
    assert not followed_review(reviewed, stored, product_id=a, as_of=now_iso())['needed_review']
    assert old['needed_review']
    workspace.store.close()
