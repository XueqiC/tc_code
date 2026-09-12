from types import SimpleNamespace
import io
import json
import urllib.error

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import appworld_teacher as shared
from bfas.mech_bfcl.common import read_rows
from bfas.mech_bfcl.teacher import Teacher, UnknownUsage, ledger_summary


@pytest.fixture
def config():
    return shared.TeacherConfig(name='openai/gpt-5.6-luna', model='gpt-5.6-luna',
        endpoint='https://api.openai.com/v1/chat/completions', api_key='stub-key',
        backend='openai_api', service_tier='flex')


def event(text='{"ok":true}', finish='stop', tokens=12):
    return dict(http_status=200, request_id='request-stub', data=dict(
        id='chat-stub', model='gpt-5.6-luna-snapshot', service_tier='flex', system_fingerprint='fp-stub',
        usage=dict(prompt_tokens=20,completion_tokens=tokens,total_tokens=20+tokens,
                   completion_tokens_details=dict(reasoning_tokens=9),prompt_tokens_details=dict(cached_tokens=5)),
        choices=[dict(finish_reason=finish,message=dict(content=text))]))


@pytest.mark.parametrize('text', ['broken JSON', '[]', '42', 'null'])
def test_discarded_generation_and_cached_attempt_still_charged(tmp_path, config, text):
    def generate(c,m,**kwargs):
        assert kwargs['retries'] == 0
        kwargs['response_callback'](event(text))
        return text
    teacher = Teacher(tmp_path, config, generate)
    value, call = teacher.ask('C','first',[],100)
    assert value is None
    teacher.ask('C','first',[],100)
    summary = ledger_summary(teacher.path)['C']
    assert summary == dict(input_tokens=20,output_tokens=12,calls=1,unresolved=0)
    records = read_rows(teacher.path)
    assert records[1]['returned_model'] == 'gpt-5.6-luna-snapshot'
    assert records[-1]['usage']['completion_tokens_details']['reasoning_tokens'] == 9


def test_timeout_reserves_and_blocks_further_purchases(tmp_path, config):
    calls = []
    def generate(*a,**k):
        calls.append(1)
        raise TimeoutError('possibly billed')
    teacher = Teacher(tmp_path,config,generate)
    with pytest.raises(UnknownUsage):
        teacher.ask('D','first',[],100)
    with pytest.raises(UnknownUsage):
        teacher.ask('C','second',[],100)
    assert len(calls) == 1
    assert ledger_summary(teacher.path)['D']['unresolved'] == 1


def test_cap_and_truncated_reasoning_usage(tmp_path, config):
    caps = []
    def generate(c,m,**k):
        caps.append(k['max_completion_tokens'])
        k['response_callback'](event('{}','length',k['max_completion_tokens']))
        return '{}'
    teacher = Teacher(tmp_path,config,generate)
    teacher.ask('shared','one',[],7000)
    teacher.ask('shared','two',[],7000)
    assert teacher.ask('shared','three',[],100) == (None,None)
    assert caps == [7000,1000]
    assert ledger_summary(teacher.path)['shared']['output_tokens'] == 8000


def test_shared_transport_reports_error_usage_before_raising(monkeypatch,config):
    response = event()['data']
    class Opener:
        def open(self, request, timeout):
            body=json.loads(request.data)
            assert body['service_tier'] == 'flex'
            assert body['model'] == 'gpt-5.6-luna'
            raise urllib.error.HTTPError(config.endpoint,500,'error',{},io.BytesIO(json.dumps(response).encode()))
    monkeypatch.setattr(shared.urllib.request,'build_opener',lambda:Opener())
    observed=[]
    with pytest.raises(shared.TeacherAPIError):
        shared.generate_reply(config,[],retries=0,response_callback=observed.append)
    assert observed[0]['data']['usage']['completion_tokens'] == 12


def test_full_reservations_use_configured_arm_budget_and_cached_calls(tmp_path, config):
    caps = []
    def generate(c, m, **kwargs):
        cap = kwargs['max_completion_tokens']
        caps.append(cap)
        kwargs['response_callback'](event('{}', tokens=cap))
        return '{}'
    teacher = Teacher(tmp_path, config, generate)
    for index in range(10):
        assert teacher.ask('C', str(index), [], 3000, output_budget=30000, require_full_cap=True)[1]
    assert teacher.ask('C', 'next', [], 3000, output_budget=30000, require_full_cap=True) == (None, None)
    assert teacher.ask('C', '0', [], 3000, output_budget=30000, require_full_cap=True)[1]
    assert caps == [3000] * 10
    assert ledger_summary(teacher.path)['C']['output_tokens'] == 30000
