"""Author-only head/base evidence; never part of the upstream patch."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


SOURCE = os.environ['SOURCE']
BASE = '551ab5bbd401b3a6b83611dedb83b04f07400057'
HEAD = 'c2476a0d02c18894f6dbd4f39f878260287a1735'
MODULE = 'tensorflow_datasets/core/dataset_info.py'
TEST = 'tensorflow_datasets/core/dataset_info_test.py'
assert SOURCE in ('base', 'head')
expected_source = subprocess.check_output([
    'git', 'show', f'{BASE if SOURCE == "base" else HEAD}:{MODULE}',
])
assert Path(MODULE).read_bytes().replace(b'\r\n', b'\n') == expected_source
expected_test = subprocess.check_output(['git', 'show', f'{HEAD}:{TEST}'])
assert Path(TEST).read_bytes().replace(b'\r\n', b'\n') == expected_test
protobuf = importlib.metadata.version('protobuf')
tf = importlib.metadata.version('tensorflow-cpu')
assert protobuf == os.environ['EXPECT_PROTOBUF']
assert tf == os.environ['EXPECT_TF']
command = [
    sys.executable, '-m', 'pytest', '-q', '--tb=short',
    '--junitxml=info-results.xml', TEST,
    'tensorflow_datasets/core/read_only_builder_test.py',
    'tensorflow_datasets/core/splits_test.py',
    'tensorflow_datasets/core/subsplits_utils_test.py',
]
result = subprocess.run(command, check=False)
cases = list(ET.parse('info-results.xml').getroot().iter('testcase'))
failures = {
    c.attrib['classname'] + '::' + c.attrib['name']: c.find('failure').attrib['message']
    for c in cases if c.find('failure') is not None
}
errors = [c.attrib for c in cases if c.find('error') is not None]
expected_failures = set()
if platform.system() == 'Windows':
  expected_failures = {
      'tensorflow_datasets.core.dataset_info_test.DatasetInfoTest::test_writing',
      'tensorflow_datasets.core.read_only_builder_test::test_get_default_config_name_permission_error',
      'tensorflow_datasets.core.read_only_builder_test::test_get_version',
      'tensorflow_datasets.core.read_only_builder_test::test_get_version_empty_builder_dir',
      'tensorflow_datasets.core.splits_test::test_multi_split_infos',
      'tensorflow_datasets.core.splits_test.SplitDictTest::test_multi_split_empty_shard',
      'tensorflow_datasets.core.splits_test.SplitDictTest::test_multi_split_sub_split',
      'tensorflow_datasets.core.splits_test.SplitsTest::test_split_file_instructions',
      'tensorflow_datasets.core.splits_test.SplitsTest::test_sub_split_file_instructions',
      'tensorflow_datasets.core.splits_test.GetDatasetFilesTest::test_no_skip_no_take',
      'tensorflow_datasets.core.splits_test.GetDatasetFilesTest::test_skip',
      'tensorflow_datasets.core.splits_test.GetDatasetFilesTest::test_skip_take1',
      'tensorflow_datasets.core.splits_test.GetDatasetFilesTest::test_skip_take2',
      'tensorflow_datasets.core.splits_test.GetDatasetFilesTest::test_take',
      'tensorflow_datasets.core.splits_test.SplitInfoTest::test_file_spec',
  }
if SOURCE == 'base' and protobuf.startswith('6.'):
  for index in range(4):
    name = (
        'tensorflow_datasets.core.dataset_info_test::'
        'test_read_from_directory_without_deprecated_label'
        f'[code_fields{index}-True]'
    )
    expected_failures.add(name)
    assert 'label() is deprecated' in failures.get(name, '')
report = {
    'source': SOURCE, 'base': BASE, 'head': HEAD,
    'os': platform.system(), 'python': platform.python_version(),
    'protobuf': protobuf, 'tensorflow': tf,
    'source_sha256': hashlib.sha256(expected_source).hexdigest(),
    'test_sha256': hashlib.sha256(expected_test).hexdigest(),
    'cases': len(cases),
    'passed': sum(not any(c.find(t) is not None for t in ('failure', 'error', 'skipped')) for c in cases),
    'skipped': sum(c.find('skipped') is not None for c in cases),
    'failures': failures, 'errors': errors,
    'expected_failures': sorted(expected_failures), 'pytest_exit': result.returncode,
}
Path('info-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('TFDS_RESTORE_RESULT=' + json.dumps(report), flush=True)
assert not errors, errors
assert len(cases) == 140, report
assert set(failures) == expected_failures, report
assert result.returncode == (1 if expected_failures else 0), result.returncode
