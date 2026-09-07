import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
import pytest

from hypercast4d.architecture import presets
from hypercast4d.compute import normalize_execution
from hypercast4d.gcp_compute import GCP_MACHINES, gcp_capability, gcp_config
from hypercast4d import gcp_runner as runner
from hypercast4d.gcp_worker import trial_requests, merge_trials, run_gpu_trials
from hypercast4d.playground_runner import run_job


@pytest.mark.parametrize('gpu,count,machine', [
    (gpu, count, machine) for gpu, machines in GCP_MACHINES.items() for count, machine in machines.items()
])
def test_machine_mapping(gpu, count, machine):
    assert normalize_execution({'target': 'gcp', 'gpu': gpu, 'gpu_count': count}) == {
        'target': 'gcp', 'gpu': gpu, 'gpu_count': count, 'machine_type': machine,
    }


@pytest.mark.parametrize('gpu,count', [('A100-40GB', 16), ('L4', 32), ('L4', 3), ('L4', True), ('L4', '2'), ('L4', 2.0), ('A100-80GB', 1)])
def test_invalid_hardware(gpu, count):
    with pytest.raises(ValueError):
        normalize_execution({'target': 'gcp', 'gpu': gpu, 'gpu_count': count})


@pytest.fixture
def config(monkeypatch):
    values = dict(PROJECT='test-project', ZONE='us-central1-a', BUCKET='test-bucket',
                  SERVICE_ACCOUNT='worker@test-project.iam.gserviceaccount.com',
                  IMAGE='prepared-image', IMAGE_PROJECT='test-project', SUBNET='private-subnet', MAX_HOURS='1')
    for key, value in values.items():
        monkeypatch.setenv('HYPERCAST_GCP_' + key, value)
    return gcp_config()


def test_capability_without_auth(config, monkeypatch):
    monkeypatch.setattr('hypercast4d.gcp_compute.shutil.which', lambda _: None)
    assert not gcp_capability()['available']


def test_bad_config(config, monkeypatch):
    monkeypatch.setenv('HYPERCAST_GCP_BUCKET', 'bucket; touch /tmp/unsafe')
    with pytest.raises(ValueError):
        gcp_config()


def test_create_has_cost_and_network_guards(config):
    args = runner.create_arguments(config, 'hypercast-test', 'g2-standard-24', Path('/tmp/startup.sh'))
    assert '--no-address' in args
    assert '--max-run-duration=1h' in args
    assert '--instance-termination-action=DELETE' in args
    assert '--boot-disk-auto-delete' in args
    assert '--no-restart-on-failure' in args
    assert '--machine-type=g2-standard-24' in args
    script = runner.startup_script('gs://test-bucket/hypercast4d/test', 2)
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
    assert 'shutdown -h' not in script
    assert '--gpu-count 2' in script


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data/raw/paper_data.xlsx'
    data.parent.mkdir(parents=True)
    values = 10 + np.cumsum(np.random.default_rng(11).normal(.01, .05, (140, 4)), axis=0)
    frame = pd.DataFrame(values, columns=['Copper', 'FCX', 'CLP', 'SCCO'])
    frame.insert(0, 'Date', pd.date_range('2020-01-01', periods=len(frame)))
    frame.to_excel(data, index=False)
    path = tmp_path / 'job'
    path.mkdir()
    request = {'phase': 'validation', 'architecture': next(p for p in presets() if p['preset_id'] == 'paper-quaternion'),
               'evaluation': {'preset': 'quick', 'epochs': 1, 'seeds': [7, 19], 'device': 'cpu'},
               'execution': {'target': 'gcp', 'gpu': 'L4', 'gpu_count': 2}}
    (path / 'request.json').write_text(json.dumps(request))
    return path


def test_payload_is_scoped_and_final_parent_relocated(job, tmp_path):
    parent = tmp_path / 'parent-original'
    parent.mkdir()
    (parent / 'runs.csv').write_text('epochs_ran\n1\n')
    request = json.loads((job / 'request.json').read_text())
    request.update(phase='final_test', parent_job_dir=str(parent))
    (job / 'request.json').write_text(json.dumps(request))
    archive_path = tmp_path / 'payload.zip'
    runner.package_job(job, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert 'parent/runs.csv' in archive.namelist()
        assert 'data/raw/paper_data.xlsx' in archive.namelist()
        assert not any('.venv' in name or '.git/' in name for name in archive.namelist())
        request = json.loads(archive.read('job/request.json'))
        assert request['parent_job_dir'] == '/opt/hypercast-job/parent'
        assert request['evaluation']['device'] == 'cuda'


@pytest.mark.parametrize('phase', ['validation', 'final_test'])
def test_sharded_results_match_sequential(job, phase):
    request = json.loads((job / 'request.json').read_text())
    if phase == 'final_test':
        parent = job / 'parent'
        parent.mkdir()
        (parent / 'request.json').write_text(json.dumps(request))
        run_job(parent)
        request.update(phase='final_test', parent_job_dir=str(parent))
        (job / 'request.json').write_text(json.dumps(request))
    run_job(job)
    expected = pd.read_csv(job / 'runs.csv')
    expected_predictions = pd.read_csv(job / 'predictions.csv')
    trials = trial_requests(request)
    assert len(trials) == 2
    paths = []
    for index, trial in enumerate(trials):
        path = job / str(index)
        path.mkdir()
        (path / 'request.json').write_text(json.dumps(trial))
        run_job(path)
        paths.append(path)
    merge_trials(job, paths)
    # Timing is inherently different; metrics, epochs and predictions must agree.
    actual = pd.read_csv(job / 'runs.csv')
    columns = [c for c in expected.columns if 'seconds' not in c]
    pd.testing.assert_frame_equal(actual[columns], expected[columns])
    pd.testing.assert_frame_equal(pd.read_csv(job / 'predictions.csv'), expected_predictions)
    assert json.loads((job / 'status.json').read_text())['completed'] == 2


def test_worker_assigns_distinct_gpu_lanes(job, monkeypatch):
    monkeypatch.setattr('torch.cuda.device_count', lambda: 2)
    seen = []
    def fake_run(args, **kwargs):
        seen.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
    monkeypatch.setattr('hypercast4d.gcp_worker.subprocess.run', fake_run)
    monkeypatch.setattr('hypercast4d.gcp_worker.merge_trials', lambda *args: None)
    run_gpu_trials(job, 2)
    assert sorted(seen) == ['0', '1']


@pytest.mark.parametrize('failure', [RuntimeError('capacity exhausted'), SystemExit(130)])
def test_cleanup_on_launch_failure_or_cancel(config, job, monkeypatch, failure):
    calls = []
    def fake_cloud(*args, **kwargs):
        calls.append(args)
        if args[:3] == ('compute', 'instances', 'create'):
            raise failure
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(runner, 'cloud', fake_cloud)
    with pytest.raises(type(failure)):
        runner.run_gcp_job(job)
    assert any(args[:3] == ('compute', 'instances', 'delete') for args in calls)
    assert any(args[:2] == ('storage', 'rm') and '/hypercast4d/hypercast-' in args[-1] for args in calls)
    assert json.loads((job / 'status.json').read_text())['gcp_cleanup'] == 'complete'
    if isinstance(failure, SystemExit):
        assert json.loads((job / 'status.json').read_text())['state'] == 'cancelled'


def test_collect_rejects_failed_worker_and_does_not_extract_paths(job, tmp_path):
    archive_path = tmp_path / 'result.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('job/exit-code', '1')
        archive.writestr('../escaped', 'unsafe')
    with pytest.raises(RuntimeError, match='worker failed'):
        runner.collect_result(archive_path, job)
    assert not (tmp_path / 'escaped').exists()


def test_successful_controller_downloads_and_cleans(config, job, monkeypatch):
    calls = []
    def fake_cloud(*args, **kwargs):
        calls.append(args)
        if args[:2] == ('storage', 'cp') and args[2].endswith('/result.zip'):
            with zipfile.ZipFile(args[3], 'w') as archive:
                for name in runner.RESULT_FILES:
                    content = json.dumps({'state': 'complete', 'completed': 2, 'total': 2}) if name == 'status.json' else 'test'
                    archive.writestr('job/' + name, content)
                archive.writestr('job/exit-code', '0')
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(runner, 'cloud', fake_cloud)
    runner.run_gcp_job(job)
    status = json.loads((job / 'status.json').read_text())
    assert status['state'] == 'complete'
    assert status['gpu_count'] == 2
    assert status['gcp_cleanup'] == 'complete'
    assert (job / 'runs.csv').read_text() == 'test'
    assert any(args[:3] == ('compute', 'instances', 'delete') for args in calls)


def test_cleanup_failure_is_visible(config, job, monkeypatch):
    def fake_cloud(*args, **kwargs):
        if args[:3] == ('compute', 'instances', 'create'):
            raise RuntimeError('launch failed')
        if args[:3] == ('compute', 'instances', 'delete'):
            return subprocess.CompletedProcess(args, 1, '', 'permission denied')
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(runner, 'cloud', fake_cloud)
    with pytest.raises(RuntimeError):
        runner.run_gcp_job(job)
    assert 'permission denied' in json.loads((job / 'status.json').read_text())['gcp_cleanup']


def test_submit_gcp_normalizes_execution_and_forces_cuda(tmp_path, monkeypatch):
    from hypercast4d.playground import JobManager
    monkeypatch.setattr('hypercast4d.playground.gcp_capability', lambda: {'available': True})
    manager = JobManager(tmp_path / 'results', tmp_path)
    job = manager.submit_validation(presets()[0], {'preset': 'quick'},
                                    {'target': 'gcp', 'gpu': 'A100-40GB', 'gpu_count': 4})
    assert job['request']['evaluation']['device'] == 'cuda'
    assert job['request']['execution']['machine_type'] == 'a2-highgpu-4g'
    assert job['status']['gpu_count'] == 4


def test_submit_gcp_requires_configuration(tmp_path, monkeypatch):
    from hypercast4d.playground import JobManager
    monkeypatch.setattr('hypercast4d.playground.gcp_capability', lambda: {'available': False, 'message': 'Setup needed'})
    manager = JobManager(tmp_path / 'results', tmp_path)
    with pytest.raises(ValueError, match='Setup needed'):
        manager.submit_validation(presets()[0], {}, {'target': 'gcp'})


def test_l4_16_uses_two_real_eight_gpu_machines():
    execution = normalize_execution({'target': 'gcp', 'gpu': 'L4', 'gpu_count': 16, 'vm_count': 99})
    assert execution == {'target': 'gcp', 'gpu': 'L4', 'gpu_count': 16,
                         'machine_type': 'g2-standard-96', 'vm_count': 2, 'gpus_per_vm': 8}


def test_vm_shards_cover_trials_exactly_once(job):
    request = json.loads((job / 'request.json').read_text())
    request['evaluation']['seeds'] = list(range(19))
    all_trials = trial_requests(request)
    first, second = trial_requests(request, 0, 2), trial_requests(request, 1, 2)
    assert first == all_trials[::2]
    assert second == all_trials[1::2]
    assert len(first) + len(second) == len(all_trials)


def test_empty_vm_shard_returns_valid_empty_artifacts(job, monkeypatch):
    request = json.loads((job / 'request.json').read_text())
    request['evaluation']['seeds'] = [7]
    (job / 'request.json').write_text(json.dumps(request))
    monkeypatch.setattr('torch.cuda.device_count', lambda: 8)
    run_gpu_trials(job, 8, 1, 2)
    assert json.loads((job / 'status.json').read_text())['completed'] == 0
    assert all((job / name).exists() for name in runner.RESULT_FILES)


@pytest.mark.parametrize('fail_second', [False, True])
def test_two_vm_controller_merges_and_cleans_both(config, job, monkeypatch, fail_second):
    request = json.loads((job / 'request.json').read_text())
    request['execution']['gpu_count'] = 16
    (job / 'request.json').write_text(json.dumps(request))
    sources = []
    for index, trial in enumerate(trial_requests(request)):
        path = job / f'prepared-{index}'
        path.mkdir()
        (path / 'request.json').write_text(json.dumps(trial))
        run_job(path)
        (path / 'training.log').write_text('test log')
        sources.append(path)
    calls = []
    def fake_cloud(*args, **kwargs):
        calls.append(args)
        if args[:3] == ('compute', 'instances', 'create'):
            assert '--machine-type=g2-standard-96' in args
            script = Path(next(arg.split('startup-script=')[1] for arg in args if 'startup-script=' in arg)).read_text()
            assert '--gpu-count 8' in script
            assert '--shard-count 2' in script
            if fail_second and args[3].endswith('-1'):
                raise RuntimeError('second VM capacity unavailable')
        if args[:2] == ('storage', 'cp') and args[2].endswith('/result.zip'):
            source = sources[1 if '/vm-1/' in args[2] else 0]
            with zipfile.ZipFile(args[3], 'w') as archive:
                for name in runner.RESULT_FILES:
                    archive.write(source / name, 'job/' + name)
                archive.writestr('job/exit-code', '0')
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(runner, 'cloud', fake_cloud)
    if fail_second:
        with pytest.raises(RuntimeError, match='second VM'):
            runner.run_gcp_job(job)
    else:
        runner.run_gcp_job(job)
        rows = pd.read_csv(job / 'runs.csv')
        assert sorted(rows.seed.tolist()) == [7, 19]
        assert json.loads((job / 'status.json').read_text())['completed'] == 2
    deleted = [args[3] for args in calls if args[:3] == ('compute', 'instances', 'delete')]
    assert len(set(deleted)) == 2
    assert json.loads((job / 'status.json').read_text())['gcp_cleanup'] == 'complete'
