import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bfas.mech_bfcl.common import ROOT
from bfas.mech_bfcl.exercises import OfficialValidator, demonstration, materialize, native_pair
from bfas.mech_bfcl.harness import pack, unpack
from bfas.mech_bfcl.pipeline import generation_prompt


def context():
    return dict(task_id='task_1', parent_id='parent', category='simple_python', frame_id='task_1:0',
        messages=[dict(role='user',content='original')], functions=[dict(name='lookup')],
        snapshot={}, snapshot_error=None, involved_classes=[])


def proposal():
    return dict(user='Find the blue item',demo=dict(kind='call',calls=[dict(name='lookup',arguments={'color':'blue'})]),
                variant='surface')


def test_materialization_whitelist_and_arm_control():
    ctx=context()
    c=materialize(proposal(),ctx,arm='C',group='c',index=0,call_id='bill')
    d=materialize(proposal(),ctx,arm='D',group='d',index=0,call_id='bill')
    assert c['functions'] == d['functions'] == ctx['functions']
    assert c['snapshot'] == d['snapshot'] == ctx['snapshot']
    assert c['messages'] == d['messages']
    assert ctx['messages'][0]['content'] == 'original'
    for extra in ['diagnosis','snapshot','functions','messages']:
        with pytest.raises(ValueError,match='Unexpected'):
            materialize(dict(proposal(),**{extra:'forged'}),ctx,arm='D',group='d',index=0,call_id='bill')
    text=json.dumps(generation_prompt(ctx,dict(hypothesis='private gap')))
    assert 'private gap' in text
    assert 'private gap' not in json.dumps(generation_prompt(ctx))


def test_natural_context_unchanged_and_no_fabricated_eos():
    ctx=context()
    p=proposal(); p.pop('user')
    row=materialize(p,ctx,arm='heldout',group='h',index=0,call_id='bill',layer=2)
    assert row['messages'] == ctx['messages']
    with pytest.raises(ValueError,match='byte-identical'):
        materialize(proposal(),ctx,arm='heldout',group='h',index=0,call_id='bill',layer=2)
    with pytest.raises(ValueError,match='end-of-task'):
        demonstration(dict(kind='abstain',text='Done<eos>'))


def test_native_local_supervision_removes_only_trailing_terminator(monkeypatch):
    stub=SimpleNamespace(render_pair=lambda *a: ('prompt', '<|tool_call>call:f{}<tool_call|><|tool_response>'))
    monkeypatch.setitem(sys.modules,'tools.bfcl_pool_render_gemma4',stub)
    row=materialize(proposal(),context(),arm='C',group='g',index=0,call_id='bill')
    prompt,target=native_pair(None,row)
    assert target == '<|tool_call>call:f{}<tool_call|>'
    assert not target.endswith('<eos>')


def test_validation_records_unexecutable_scope_and_execution_failure():
    v=OfficialValidator.__new__(OfficialValidator)
    v.ast=lambda *a: {'valid':True}
    v.execute=lambda *a: None
    row=materialize(proposal(),context(),arm='C',group='g',index=0,call_id='bill')
    check=v.validate(row)
    assert check['valid'] and not check['executable_consistency']
    assert any('No official executable' in s for s in check['unvalidated'])
    v.execute=lambda *a: (['Error during execution: bad state'],{})
    assert not v.validate(row)['valid']


def test_snapshot_types_roundtrip():
    value=dict(tuple_key={(1,2): {'a','b'}},path=ROOT / 'relative',values=[None,True,1,2.3])
    assert unpack(json.loads(json.dumps(pack(value)))) == value
    with pytest.raises(TypeError):
        pack(object())


def test_official_checker_and_handler_registration_cpu(tmp_path):
    """Real checkout checkers with stub responses; no network, tokenizer or GPU."""
    python=ROOT/'envs/bfcl/.venv/bin/python'
    if not python.exists():
        pytest.skip('BFCL environment not installed')
    code = '''
from bfas.mech_bfcl.common import setup_harness
setup_harness()
from bfas.mech_bfcl.harness import register_capture
register_capture()
from bfas.mech_bfcl.exercises import OfficialValidator
v=OfficialValidator()
schema=dict(name='add',parameters=dict(type='dict',properties={'x':{'type':'integer'}},required=['x']))
e=dict(category='simple_python',functions=[schema],demo={'kind':'call','calls':[{'name':'add','arguments':{'x':2}}]},involved_classes=[])
assert v.validate(e)['valid']
assert v.score(e,'<|tool_call>call:add{x:2}<tool_call|>')['correct']
assert not v.score(e,'<|tool_call>call:add{x:3}<tool_call|>')['correct']
assert not v.score(e,'<|tool_call>call:invented{x:2}<tool_call|>')['correct']
assert not v.score(e,'<|tool_call>call:add{}<tool_call|>')['correct']
e['demo']={'kind':'abstain','text':'Please provide x.'}
assert v.score(e,'Please provide x.')['correct']
assert not v.score(e,'<|tool_call>bad')['correct']
from types import SimpleNamespace
from pathlib import Path
import os
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfas.mech_bfcl.common import REGISTRY, STUDENT
from bfas.mech_bfcl.harness import frames
class Tokenizer:
    def apply_chat_template(self,*a,**k):
        assert k['enable_thinking'] is False
        return 'native prompt'
    def encode(self,*a,**k): return [1,2,3]
    def convert_tokens_to_ids(self,t): return 4
class Client:
    def __init__(self): self.completions=self
    def with_options(self,**k):
        assert k['max_retries']==0
        return self
    def create(self,**k):
        assert k['model']==STUDENT and k['extra_body']['top_k']==1 and k['temperature']==.001
        usage=SimpleNamespace(prompt_tokens=3,completion_tokens=6,model_dump=lambda:{'prompt_tokens':3,'completion_tokens':6})
        return SimpleNamespace(choices=[SimpleNamespace(text='<|tool_call>call:add{x:2}<tool_call|>',finish_reason='stop')],usage=usage)
capture=Path(os.environ['BFCL_PROJECT_ROOT'])/'capture'
os.environ.update(MECH_CAPTURE_DIR=str(capture),MECH_SERVED_MODEL=STUDENT)
cls=MODEL_CONFIG_MAPPING[REGISTRY].model_handler
h=cls.__new__(cls)
h.tokenizer=Tokenizer(); h.client=Client(); h.temperature=.001; h.model_path_or_id=STUDENT
result,metadata=h.inference(dict(id='simple_python_0',question=[[{'role':'user','content':'Add two'}]],function=[schema]),True,False)
captured=frames(capture)
assert len(captured)==1 and captured[0]['response']==result
assert captured[0]['messages']==[{'role':'user','content':'Add two'}]
print('official CPU checks passed')
'''
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),BFCL_PROJECT_ROOT=str(tmp_path/'runtime'),
             CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    result=subprocess.run([str(python),'-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
