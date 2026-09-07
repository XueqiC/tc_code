"""Protocol 1.0.7 projections of mixed scoring/operational source files.

These are source projections, never imports or executions of audited code.
The exact selectors and excluded device-placement keywords are protocol, not
an automatically inferred dependency graph. See docs/rtd_v1_protocol.md.
"""
import ast
import hashlib
import re

from .persistence import digest


PYTHON_SCOPES = {
    'tools/behavior_atom/checker_bridge.py': (
        'ROOT', 'BFCL', '_load_direct', '_content_hash', '_checker_version',
        '_language_name', '_check_multi_turn', '_check_relevance',
        'CheckerBridge.__init__', 'CheckerBridge.check', 'CheckerBridge.__call__',
        'CheckerBridge.check_multi_turn', 'CheckerBridge.check_relevance',
        'CheckerBridge.check_many', 'CheckerBridge._request'),
    'tools/bfcl_event_mine_single.py': ('language_for',),
    'tools/bfcl_generation_check.py': ('check_generation',),
    'src/bfas/rtd/evaluation.py': ('official_expectations', 'validate_evaluation', '_flatten_adapter'),
    'src/bfas/adapters/bfcl.py': ('BFCLScoreError', '_score_category', '_generated_by_category',
                               'read_score_summaries', 'extract_verdicts'),
}
SHELL = 'tools/bfcl_std_campaign.sh'

# C25m's only representation equivalence: percent suffix/whitespace and N/A.
# Match the entire helper AST before normalizing its call to historical float;
# any other helper implementation remains scoring content (including scaling).
_PERCENTAGE_PARSER = """
def _parse_percentage(value):
    value = value.strip().removesuffix('%').strip()
    return None if value == 'N/A' else float(value)
"""


class _ScientificAST(ast.NodeTransformer):
    def __init__(self, *, placement=False, diagnostics=False):
        self.placement, self.diagnostics = placement, diagnostics

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # docstrings
        if (self.diagnostics and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
                and any(k.arg == 'file' and ast.unparse(k.value) == 'sys.stderr'
                        for k in node.value.keywords)):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (self.placement and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'from_pretrained'):
            node.keywords = [k for k in node.keywords if k.arg not in {'device_map', 'torch_device'}]
        return self.generic_visit(node)


def scoring_projection(name, content):
    """Return a portable hash input; fail closed on missing/ambiguous selectors."""
    if name == SHELL:
        # Arguments that choose handler, weights, sampling, category population,
        # evaluator and aggregate. Wrappers, endpoints, retries and logs excluded.
        patterns = {
            'model_directory': (r'^MODEL_DIR=.*$', 1),
            'generation_arguments': (r'^  generate_args=\([\s\S]*?\)\n', 1),
            'local_weights': (r'^    generate_args\+=\(.*\)$', 1),
            'sampler': (r'^\s*VLLM_USE_FLASHINFER_SAMPLER=.*?\\$', 2),
            'generate': (r'^\s*"\$BFCL" generate .*?\\$', 2),
            'evaluate': (r'^  if ! "\$BFCL" evaluate .*?\\\n    --result-dir .*?\\$', 1),
            'aggregate': (r'^  overall=\$\(python3 -c .*?\\\n    "\$out/data_overall.csv"', 1),
        }
        result = {}
        for label, (pattern, count) in patterns.items():
            matches = re.findall(pattern, content, re.M)
            if len(matches) != count:
                raise ValueError(f'evaluation harness scope {name}:{label}: expected {count} selections')
            if label == 'generation_arguments':
                matches = [re.sub(r'--(?:num-gpus|num-threads|gpu-memory-utilization|result-dir) '
                                  r'(?:"[^"\n]*"|[^\s)]+)', '', s) for s in matches]
            result[label] = ['\n'.join(line.strip() for line in s.strip().splitlines() if line.strip()) for s in matches]
        return result
    if name not in PYTHON_SCOPES:
        return None
    tree = ast.parse(content, filename=name)
    selected = []
    for symbol in PYTHON_SCOPES[name]:
        body = tree.body
        parts = symbol.split('.')
        if len(parts) == 2:
            classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == parts[0]]
            if len(classes) != 1:
                raise ValueError(f'evaluation harness scope {name}:{symbol}: expected one class')
            body = classes[0].body
        nodes = [n for n in body if
                 (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == parts[-1]) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == parts[-1]
                                                   for t in n.targets))]
        if len(nodes) != 1:
            raise ValueError(f'evaluation harness scope {name}:{symbol}: expected one definition')
        selected.append(_ScientificAST(placement=symbol == '_flatten_adapter',
                                      diagnostics=name.endswith('bfcl_generation_check.py')).visit(nodes[0]))
    if name == 'tools/behavior_atom/checker_bridge.py':
        # Pin worker dispatch, boolean validation and envelope contents, while
        # excluding line IO/serialization and the diagnostic repr fallback.
        workers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker_main']
        blocks = [n for worker in workers for n in ast.walk(worker) if isinstance(n, ast.Try)
                  and any(isinstance(stmt, ast.Assign) and any(
                      isinstance(t, ast.Name) and t.id == 'response' for t in stmt.targets)
                          for stmt in n.body)]
        if len(workers) != 1 or len(blocks) != 1:
            raise ValueError('evaluation harness scope checker_bridge.py: expected one worker verdict block')
        selected.append(_ScientificAST().visit(blocks[0]))
    if name == 'src/bfas/rtd/evaluation.py':
        # Score-reading and external Hub overlay invocation also affect results.
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate']
        if len(functions) != 1:
            raise ValueError('evaluation harness scope requires evaluate')
        nodes = list(ast.walk(functions[0]))
        for label, predicate in (
            ('aggregate', lambda n: isinstance(n, ast.With)
             and any('data_overall.csv' in ast.unparse(i.context_expr) for i in n.items)),
            ('aggregate_range', lambda n: isinstance(n, ast.If) and 'math.isfinite(score)' in ast.unparse(n.test)),
            ('merge_command', lambda n: isinstance(n, ast.Call) and ast.unparse(n.func) == 'subprocess.run'
             and n.args and 'tools/bfcl_hub_merge_export.py' in ast.unparse(n.args[0])),
        ):
            matches = [n for n in nodes if predicate(n)]
            if len(matches) != 1:
                raise ValueError(f'evaluation harness scope evaluation.py:{label}: expected one selection')
            if label == 'aggregate':
                calls = [n for n in ast.walk(matches[0]) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == '_parse_percentage']
                if calls:
                    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name == '_parse_percentage']
                    if len(helpers) != 1:
                        raise ValueError('evaluation harness scope requires one percentage parser')
                    helper = _ScientificAST().visit(helpers[0])
                    if ast.dump(helper) == ast.dump(ast.parse(_PERCENTAGE_PARSER).body[0]):
                        for call in calls:
                            call.func.id = 'float'
                    else:
                        selected.append(helper)
            if label == 'aggregate_range':
                test = matches[0].test
                if (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                        and len(test.values) == 2 and ast.unparse(test.values[0]) == 'score is not None'):
                    matches[0].test = test.values[1]
            # Working directory and CPU environment are placement metadata.
            selected.append(matches[0].args[0] if label == 'merge_command' else matches[0])
    # Bind the module imports actually referenced by the selected code, so
    # changing an import alias cannot silently change a projected function.
    used = {n.id for item in selected for n in ast.walk(item) if isinstance(n, ast.Name)}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a for a in node.names if (a.asname or a.name.split('.')[0]) in used]
            if names:
                node.names = names
                imports.append(node)
    return [ast.dump(n, include_attributes=False) for n in imports + selected]


def scoring_hash(name, content):
    projection = scoring_projection(name, content)
    return (hashlib.sha256(content.encode()).hexdigest() if projection is None else digest(projection))
