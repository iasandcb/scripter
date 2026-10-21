"""Pull mark-vector video jobs, render them, and upload the result.

The queue is pull-based: mark-vector never calls this machine. Idle polls do
not hold a sleep assertion, so the Mac can sleep. A job holds `caffeinate -i`
until it finishes. Progress renews the lease. If the lease expires the job is
requeued; a video id already stored locally is reported again and not uploaded
twice.

Contract (mark-vector "YouTube 영상 작업 큐"):
  Authorization: Bearer $WORKER_TOKEN
  GET  /api/worker/jobs?order=oldest|newest&status=...
  POST /api/worker/jobs/claim          {worker_id, lease_seconds}
  GET  /api/worker/jobs/{id}/audio     X-Claim-Token
  POST /api/worker/jobs/{id}/progress  {claim_token, message?, percent?, lease_seconds?}
  POST /api/worker/jobs/{id}/complete  {claim_token, youtube_video_id, youtube_url?, transcript?}
  POST /api/worker/jobs/{id}/fail      {claim_token, error}
"""
import json
import os
import plistlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import youtube_upload
from net import atomic_write, error_text, header, request, retry_after


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = 'https://markvector.fly.dev'
TEMPLATE = ROOT / 'macos' / 'com.scripter.worker.plist'
LEASE_LOST = frozenset({401, 403, 404, 409, 410})
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MIME_EXT = {
    'audio/mp4': '.m4a', 'audio/m4a': '.m4a', 'audio/x-m4a': '.m4a', 'audio/aac': '.aac',
    'audio/mpeg': '.mp3', 'audio/mp3': '.mp3', 'audio/wav': '.wav', 'audio/x-wav': '.wav',
    'audio/flac': '.flac', 'audio/ogg': '.ogg', 'audio/aiff': '.aiff', 'audio/x-aiff': '.aiff',
    'video/mp4': '.mp4',
}


class WorkerError(RuntimeError):
    pass


class AuthError(WorkerError):
    """WORKER_TOKEN was rejected. Claiming again will not help until it changes."""


class LeaseLost(WorkerError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class CompletePending(WorkerError):
    """The video is uploaded. complete has not been acknowledged."""


class CompleteBlocked(WorkerError):
    """complete failed too many times. The local video id is kept."""


def log(message):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)


def clip(text, limit=1500):
    text = ' '.join(str(text).split())
    return text[:limit]


def safe_component(value):
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    if not text:
        raise WorkerError(f'작업 id가 비어 있습니다: {value!r}')
    return text[:80]


def audio_extension(mime, disposition=None):
    if disposition:
        matched = re.search(r"filename\*=UTF-8''([^;]+)", disposition) or re.search(r'filename="?([^";]+)"?', disposition)
        if matched:
            name = urllib.parse.unquote(matched.group(1))
            suffix = Path(name).suffix.lower()
            if suffix in {'.m4a', '.mp3', '.wav', '.aac', '.flac', '.ogg', '.aiff', '.mp4'}:
                return suffix
    kind = str(mime or '').split(';', 1)[0].strip().lower()
    return MIME_EXT.get(kind, '.m4a')


def base_url(value=None):
    chosen = (value or os.environ.get('WORKER_BASE_URL') or DEFAULT_BASE).strip().rstrip('/')
    if not chosen.startswith(('https://', 'http://')):
        raise WorkerError('WORKER_BASE_URL은 http(s) 주소여야 합니다.')
    return chosen


def worker_identifier(value=None):
    chosen = (value or os.environ.get('WORKER_ID') or '').strip()
    if chosen:
        return chosen
    return 'scripter-' + socket.gethostname().split('.')[0]


def heartbeat_interval(progress_seconds, lease_seconds):
    if progress_seconds <= 0:
        raise WorkerError('--progress-seconds는 양수여야 합니다.')
    if progress_seconds > lease_seconds / 3:
        return max(10, lease_seconds / 3)
    return progress_seconds


def interpret_progress(line, message, percent):
    """Map a pipeline log line onto a lease progress report."""
    if '교정' in line:
        return '전사문 교정 중', max(percent, 38)
    if '유튜브' in line or '썸네일' in line or '챕터' in line:
        return '제목·설명·썸네일 생성 중', max(percent, 82)
    matched = re.search(r'\(([\d.]+)%\)', line)
    if matched and ('인식' in line or '렌더링' in line):
        value = float(matched.group(1))
        if '렌더링' in line:
            return '영상 렌더링 중', max(percent, int(45 + value * 0.3))
        return '음성 인식 중', max(percent, int(5 + value * 0.3))
    return message, percent


class Status:
    def __init__(self):
        self._lock = threading.Lock()
        self.message = '시작'
        self.percent = 0

    def update(self, message=None, percent=None):
        with self._lock:
            if message:
                self.message = clip(message, 300)
            if percent is not None:
                self.percent = max(self.percent, min(100, int(percent)))

    def snapshot(self):
        with self._lock:
            return self.message, self.percent


class WorkerClient:
    def __init__(self, token, base, worker_id, lease_seconds, transport=request,
                 sleep=time.sleep, retries=4, timeout=120):
        self.token = token
        self.base = base.rstrip('/')
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.transport = transport
        self.sleep = sleep
        self.retries = retries
        self.timeout = timeout

    def url(self, path):
        return self.base + path

    def _request(self, method, url, body=None, claim_token=None, timeout=None):
        headers = {
            'Authorization': 'Bearer ' + self.token,
            'Accept': 'application/json',
        }
        data = None
        if claim_token:
            headers['X-Claim-Token'] = claim_token
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        last = None
        for attempt in range(self.retries):
            try:
                status, hdrs, raw = self.transport(method, url, data=data, headers=headers,
                                                   timeout=timeout or self.timeout)
            except (TimeoutError, OSError) as error:
                last = error
                if attempt + 1 == self.retries:
                    raise WorkerError(f'네트워크 오류: {error}') from error
                self.sleep(min(30, 2 ** attempt))
                continue
            if status in RETRYABLE and attempt + 1 < self.retries:
                self.sleep(retry_after(hdrs) or min(30, 2 ** attempt))
                continue
            return status, raw
        raise WorkerError(f'네트워크 오류: {last}')

    def _job_status(self, status, raw, not_found_message='작업 임대가 끝났거나 작업을 찾을 수 없습니다.'):
        if status in (200, 201, 204):
            return
        text = error_text(raw) or not_found_message
        if status == 401 and 'authenticated' in text.lower():
            raise AuthError(text)
        if status in LEASE_LOST:
            raise LeaseLost(text, status=status)
        raise WorkerError(f'HTTP {status}: {text}')

    def list_jobs(self, order='oldest', status=None):
        if order not in ('oldest', 'newest'):
            raise WorkerError('order는 oldest 또는 newest여야 합니다.')
        query = {'order': order}
        if status:
            query['status'] = status
        code, raw = self._request('GET', self.url('/api/worker/jobs') + '?' + urllib.parse.urlencode(query))
        if code == 401:
            raise AuthError(error_text(raw) or 'Not authenticated')
        self._job_status(code, raw)
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except (UnicodeError, ValueError) as error:
            raise WorkerError('작업 목록이 JSON이 아닙니다.') from error
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ('jobs', 'items'):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise WorkerError('작업 목록 형식이 잘못되었습니다.')

    def claim(self):
        code, raw = self._request('POST', self.url('/api/worker/jobs/claim'), {
            'worker_id': self.worker_id,
            'lease_seconds': self.lease_seconds,
        })
        if code == 401:
            raise AuthError(error_text(raw) or 'Not authenticated')
        self._job_status(code, raw)
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except (UnicodeError, ValueError) as error:
            raise WorkerError('claim 응답이 JSON이 아닙니다.') from error
        if not isinstance(payload, dict) or 'job' not in payload:
            raise WorkerError('claim 응답에 job이 없습니다.')
        job = payload['job']
        if job is None:
            return None
        if not isinstance(job, dict) or 'id' not in job or not job.get('claim_token'):
            raise WorkerError('claim 응답에 id 또는 claim_token이 없습니다.')
        job = dict(job)
        job['id'] = str(job['id'])
        return job

    def download_audio(self, job, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(destination.parent, 0o700)
        except OSError:
            pass
        url = self.url(f"/api/worker/jobs/{urllib.parse.quote(str(job['id']), safe='')}/audio")
        temporary = destination.with_name(destination.name + '.part')
        last = None
        for attempt in range(self.retries):
            try:
                status, hdrs, raw = self.transport(
                    'GET', url, data=None,
                    headers={
                        'Authorization': 'Bearer ' + self.token,
                        'X-Claim-Token': job['claim_token'],
                        'Accept': '*/*',
                    },
                    timeout=max(self.timeout, 300))
            except (TimeoutError, OSError) as error:
                last = error
                if attempt + 1 == self.retries:
                    raise WorkerError(f'음성을 받지 못했습니다: {error}') from error
                self.sleep(min(30, 2 ** attempt))
                continue
            if status in RETRYABLE and attempt + 1 < self.retries:
                self.sleep(min(30, 2 ** attempt))
                continue
            if status != 200:
                self._job_status(status, raw, not_found_message='음성을 받지 못했습니다.')
            if not raw:
                raise WorkerError('받은 음성이 비어 있습니다.')
            length = header(hdrs, 'Content-Length')
            expected = int(length) if length and str(length).isdigit() else None
            if expected is not None and expected != len(raw):
                raise WorkerError('받은 음성 길이가 맞지 않습니다.')
            temporary.write_bytes(raw)
            os.replace(temporary, destination)
            return
        raise WorkerError(f'음성을 받지 못했습니다: {last}')

    def progress(self, job, message=None, percent=None):
        body = {'claim_token': job['claim_token'], 'lease_seconds': self.lease_seconds}
        if message:
            body['message'] = clip(message, 300)
        if percent is not None:
            body['percent'] = max(0, min(100, int(percent)))
        code, raw = self._request(
            'POST', self.url(f"/api/worker/jobs/{urllib.parse.quote(str(job['id']), safe='')}/progress"), body,
            claim_token=job['claim_token'])
        self._job_status(code, raw)

    def complete(self, job, youtube_video_id, youtube_url, transcript):
        body = {
            'claim_token': job['claim_token'],
            'youtube_video_id': youtube_video_id,
            'youtube_url': youtube_url,
            'transcript': transcript or '',
        }
        code, raw = self._request(
            'POST', self.url(f"/api/worker/jobs/{urllib.parse.quote(str(job['id']), safe='')}/complete"), body,
            claim_token=job['claim_token'], timeout=180)
        if code in (200, 201, 204):
            return
        text = error_text(raw)
        lowered = text.lower()
        if code == 409 and any(word in lowered or word in text for word in ('already', 'completed', '완료', 'duplicate')):
            return
        self._job_status(code, raw)

    def fail(self, job, error):
        body = {'claim_token': job['claim_token'], 'error': clip(error)}
        code, raw = self._request(
            'POST', self.url(f"/api/worker/jobs/{urllib.parse.quote(str(job['id']), safe='')}/fail"), body,
            claim_token=job['claim_token'])
        self._job_status(code, raw)


def load_state(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise WorkerError(f'상태 파일을 읽을 수 없습니다: {path} ({error})') from error
    if not isinstance(data, dict):
        raise WorkerError(f'상태 파일이 잘못되었습니다: {path}')
    return data


def save_state(path, state):
    state = dict(state)
    state['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2) + '\n')


def pipeline_command(audio, output, options):
    """The existing CLI: glossary (default file), --correct, render, --youtube."""
    command = [
        sys.executable, str(ROOT / 'scripter.py'),
        '-i', str(audio), '-o', str(output),
        '--correct', '--youtube', '--allow-missing-thumbnail',
        '--engine', options.engine, '--language', options.language,
        '--youtube-model', options.youtube_model,
        '--youtube-image-model', options.youtube_image_model,
        '--correct-model', options.correct_model,
        '--correct-threshold', str(options.correct_threshold),
        '--width', str(options.width), '--height', str(options.height),
        '--fps', str(options.fps), '--font', str(options.font),
        '--font-size', str(options.font_size), '--margin', str(options.margin),
        '--lines', str(options.lines),
        '--spoken-math-url', options.spoken_math_url,
    ]
    if options.correct_force:
        command.append('--correct-force')
    if options.model:
        command += ['--model', options.model]
    if options.glossary is not None:
        command += ['--glossary', options.glossary]
    if options.spoken_math is not None:
        command += ['--spoken-math', str(options.spoken_math)]
    return command


def terminate_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait(timeout=5)


def default_pipeline(audio, output, cancel, status, options, log=log):
    command = pipeline_command(audio, output, options)
    log('실행: ' + ' '.join(shlex.quote(part) for part in command))
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    tail = []
    code = 1
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            log(text)
            tail.append(text)
            del tail[:-40]
            message, percent = interpret_progress(text, *status.snapshot())
            status.update(message=message, percent=percent)
            if cancel.is_set():
                terminate_group(proc)
                raise LeaseLost('임대가 끝나 영상 생성을 멈췄습니다.')
        code = proc.wait()
    finally:
        if proc.poll() is None:
            terminate_group(proc)
    if cancel.is_set():
        raise LeaseLost('임대가 끝나 영상 생성을 멈췄습니다.')
    if code != 0:
        detail = '\n'.join(tail[-20:])
        raise RuntimeError(f'영상 생성 프로세스가 끝났습니다 (exit {code}).\n{detail}')


def require_outputs(output):
    missing = []
    for path in (output, output.with_suffix('.txt'), output.with_suffix('.youtube.json')):
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(path.name)
    if missing:
        raise RuntimeError('영상 생성 결과가 없습니다: ' + ', '.join(missing))


def report(client, job, message, percent, status=None):
    if status is not None:
        status.update(message=message, percent=percent)
    client.progress(job, message=message, percent=percent)


def _beat(client, job, status, cancel, stop, interval, log):
    while not stop.is_set() and not cancel.is_set():
        message, percent = status.snapshot()
        try:
            client.progress(job, message=message, percent=percent)
        except LeaseLost as error:
            log(f'임대가 끝났습니다: {error}')
            cancel.set()
            return
        except AuthError as error:
            log(f'워커 인증이 거부되었습니다: {error}')
            cancel.set()
            return
        except Exception as error:
            log(f'진행 보고 실패: {error}')
        if stop.wait(interval):
            return


def acknowledge(client, job, state, path, sleep, attempts, log):
    last = None
    for attempt in range(attempts):
        try:
            client.complete(job, state['youtube_video_id'], state.get('youtube_url'), state.get('transcript') or '')
        except LeaseLost:
            save_state(path, state)
            raise
        except AuthError:
            raise
        except Exception as error:
            last = error
            state['complete_failures'] = int(state.get('complete_failures') or 0) + 1
            save_state(path, state)
            log(f'완료 보고 실패 ({state["complete_failures"]}): {error}')
            if state['complete_failures'] >= 8:
                state['phase'] = 'complete_blocked'
                save_state(path, state)
                raise CompleteBlocked(
                    f'완료 보고를 멈췄습니다. 영상은 업로드되어 있습니다: {state.get("youtube_url")}') from error
            try:
                client.progress(job, message='완료 보고 재시도', percent=99)
            except LeaseLost:
                raise
            except Exception as progress_error:
                log(f'완료 보고 중 진행 갱신 실패: {progress_error}')
            if attempt + 1 < attempts:
                sleep(min(30, 2 ** attempt))
            continue
        state['phase'] = 'completed'
        state['complete_failures'] = 0
        save_state(path, state)
        log(f'완료 보고: {state.get("youtube_url")}')
        return
    raise CompletePending(last or '완료 보고가 끝나지 않았습니다.')


def iter_states(root):
    root = Path(root)
    if not root.is_dir():
        return
    for path in sorted(root.glob('*/state.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            log(f'상태 파일을 읽지 못했습니다 ({path}): {error}')
            continue
        if isinstance(data, dict):
            yield path, data


def retry_pending(client, root, sleep, attempts, log):
    for path, state in iter_states(root):
        if state.get('phase') != 'uploaded' or not state.get('youtube_video_id') or not state.get('claim_token'):
            continue
        job = {'id': state.get('job_id') or path.parent.name, 'claim_token': state['claim_token']}
        log(f'완료 보고를 다시 시도합니다: {job["id"]} {state["youtube_video_id"]}')
        try:
            acknowledge(client, job, state, path, sleep, attempts, log)
        except LeaseLost as error:
            log(f'완료 보고의 임대가 끝났습니다. 같은 작업을 다시 받으면 이어서 보고합니다: {error}')
        except CompletePending as error:
            log(f'완료 보고가 아직 되지 않았습니다: {error}')
        except CompleteBlocked as error:
            log(str(error))


def process_job(client, job, root, options, pipeline, upload, log, sleep, progress_interval, complete_attempts):
    if not job.get('id') or not job.get('claim_token'):
        raise WorkerError('작업에 id 또는 claim_token이 없습니다.')
    folder = Path(root) / safe_component(job['id'])
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(folder, 0o700)
    except OSError:
        pass
    path = folder / 'state.json'
    state = load_state(path) or {}
    state['job_id'] = str(job['id'])
    state['claim_token'] = job['claim_token']
    state['recording_id'] = job.get('recording_id')
    state['title'] = job.get('title')
    state['mime'] = job.get('mime')
    extension = audio_extension(job.get('mime'))
    audio = Path(state['audio']) if state.get('audio') else folder / ('audio' + extension)
    output = Path(state['output']) if state.get('output') else folder / 'video.mp4'
    state['audio'] = str(audio)
    state['output'] = str(output)
    if state.get('phase') == 'completed' and state.get('youtube_video_id'):
        acknowledge(client, job, state, path, sleep, complete_attempts, log)
        return state
    if state.get('youtube_video_id'):
        log(f'이미 업로드된 영상입니다 ({state["youtube_video_id"]}). 완료만 다시 보고합니다.')
        state['phase'] = 'uploaded'
        save_state(path, state)
        acknowledge(client, job, state, path, sleep, complete_attempts, log)
        return state
    if not audio.is_file() or audio.stat().st_size <= 0:
        report(client, job, '음성 받는 중', 1)
        client.download_audio(job, audio)
        state['phase'] = 'downloaded'
        save_state(path, state)
    if state.get('phase') != 'pipeline_done':
        report(client, job, '전사·교정·영상 생성 중', 5)
        cancel = threading.Event()
        stop = threading.Event()
        status = Status()
        status.update(message='전사·교정·영상 생성 중', percent=5)
        thread = threading.Thread(
            target=_beat, name='scripter-lease', daemon=True,
            args=(client, job, status, cancel, stop, progress_interval, log))
        thread.start()
        try:
            pipeline(audio, output, cancel, status, options, log)
        finally:
            stop.set()
            thread.join(timeout=2)
        if cancel.is_set():
            raise LeaseLost('작업 중 임대가 끝났습니다. 영상은 올리지 않습니다.')
        require_outputs(output)
        state['transcript'] = output.with_suffix('.txt').read_text(encoding='utf-8')
        state['phase'] = 'pipeline_done'
        save_state(path, state)
    else:
        require_outputs(output)
        if not state.get('transcript'):
            state['transcript'] = output.with_suffix('.txt').read_text(encoding='utf-8')
    report(client, job, '유튜브 업로드 중', 90)

    def on_progress(percent):
        report(client, job, '유튜브 업로드 중', percent)

    result = upload(output, on_progress)
    state['youtube_video_id'] = result['youtube_video_id']
    state['youtube_url'] = result['youtube_url']
    state['phase'] = 'uploaded'
    if not state.get('transcript') and output.with_suffix('.txt').is_file():
        state['transcript'] = output.with_suffix('.txt').read_text(encoding='utf-8')
    save_state(path, state)
    if result.get('thumbnail') is False:
        log('썸네일은 올리지 못했습니다. 영상은 비공개로 올라가 있어 작업을 완료로 보고합니다.')
    acknowledge(client, job, state, path, sleep, complete_attempts, log)
    return state


def _finish_outcome(outcome, once):
    if once and outcome not in ('empty', 'completed'):
        raise SystemExit(1)


def run_loop(client, root, options, poll_seconds, progress_interval, once=False, pipeline=None,
             upload=None, sleep=time.sleep, log=log, credentials_ready=lambda: True, stop=None,
             complete_attempts=5):
    """Poll until stopped. `once` handles at most one job."""
    pipeline = pipeline or default_pipeline
    upload = upload or (lambda output, on_progress=None: (_ for _ in ()).throw(
        RuntimeError('upload가 연결되지 않았습니다.')))
    stop = stop or threading.Event()
    while not stop.is_set():
        retry_pending(client, root, sleep, complete_attempts, log)
        if stop.is_set():
            return
        if not credentials_ready():
            log('유튜브 인증이 없어 작업을 받지 않습니다. 터미널에서 uv run scripter.py --youtube-auth 를 실행하세요.')
            if once:
                raise SystemExit(1)
            sleep(poll_seconds)
            continue
        try:
            job = client.claim()
        except AuthError as error:
            log(f'WORKER_TOKEN이 거부되었습니다. .env를 확인하세요. ({error})')
            if once:
                raise SystemExit(1) from error
            sleep(max(poll_seconds, 300))
            continue
        except Exception as error:
            log(f'큐에 연결하지 못했습니다: {error}')
            if once:
                raise SystemExit(1) from error
            sleep(poll_seconds)
            continue
        if job is None:
            log('대기 중인 작업이 없습니다.')
            if once:
                return
            sleep(poll_seconds)
            continue
        title = job.get('title') or ''
        log(f'작업 시작: {job["id"]} {title}'.rstrip())
        outcome = 'completed'
        try:
            with keep_awake():
                process_job(client, job, root, options, pipeline, upload, log, sleep,
                            progress_interval, complete_attempts)
        except LeaseLost as error:
            outcome = 'lease'
            log(f'임대가 끝나 이 작업은 큐로 돌아갑니다. 올려 둔 영상은 다시 올리지 않습니다: {error}')
        except CompletePending as error:
            outcome = 'pending'
            log(f'업로드는 끝났고 완료 보고만 남았습니다. 다음 주기에 다시 보고합니다: {error}')
        except CompleteBlocked as error:
            outcome = 'blocked'
            log(str(error))
        except Exception as error:
            outcome = 'failed'
            log(f'작업 실패: {error}')
            try:
                client.fail(job, clip(error))
                log('실패를 큐에 보고했습니다.')
            except LeaseLost as fail_error:
                log(f'실패 보고 전에 임대가 끝났습니다: {fail_error}')
            except Exception as fail_error:
                log(f'실패 보고 실패: {fail_error}')
        if once:
            _finish_outcome(outcome, True)
            return
        if outcome != 'completed':
            sleep(poll_seconds)


def keep_awake():
    """Prevent idle sleep for the duration of the with-block. No-op off macOS."""
    return _Awake()


class _Awake:
    def __enter__(self):
        self.proc = None
        if sys.platform != 'darwin' or not shutil.which('caffeinate'):
            return self
        self.proc = subprocess.Popen(['caffeinate', '-i'])
        return self

    def __exit__(self, *exc):
        proc = getattr(self, 'proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return False


def install_stop_signals(stop):
    def handle(signum, frame):
        log('종료 신호를 받았습니다. 대기 중이면 멈추고, 작업 중이면 그 작업이 끝난 뒤 멈춥니다.')
        stop.set()
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def serve(args, install_signals=True):
    if not args.font.is_file():
        raise SystemExit(f'한글 폰트가 없습니다: {args.font}')
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise SystemExit(f'{tool}가 필요합니다.')
    if shutil.which('node') is None or not (ROOT / 'node_modules').is_dir():
        log('Node.js 또는 node_modules가 없습니다. 수식은 조판되지 않고 말로 남습니다.')
    token = os.environ.get('WORKER_TOKEN', '').strip()
    if not token:
        raise SystemExit('WORKER_TOKEN이 필요합니다. .env를 확인하세요.')
    url = base_url(args.worker_url)
    identifier = worker_identifier(args.worker_id)
    lease = args.lease_seconds
    progress = args.progress_seconds if args.progress_seconds is not None else float(os.environ.get('WORKER_PROGRESS_SECONDS', '120'))
    interval = heartbeat_interval(progress, lease)
    if interval != progress:
        log(f'진행 보고 간격을 임대({lease}초)에 맞춰 {interval:.0f}초로 줄입니다.')
    poll = args.poll_seconds if args.poll_seconds is not None else float(os.environ.get('WORKER_POLL_SECONDS', '60'))
    root = Path(args.worker_dir) if args.worker_dir is not None else ROOT / 'files' / 'worker'
    token_file = youtube_upload.resolve_token_file(args.token_file)
    if poll <= 0:
        raise SystemExit('--poll-seconds는 양수여야 합니다.')

    def ready():
        try:
            secret_path = youtube_upload.resolve_client_secret(args.client_secret)
        except RuntimeError as error:
            log(str(error))
            return False
        credentials = youtube_upload.Credentials(secret_path, token_file)
        if not credentials.ready():
            return False
        try:
            credentials.authorization()
        except Exception as error:
            log(f'유튜브 인증을 확인하지 못했습니다: {error}')
            return False
        return True

    def upload(output, on_progress=None):
        return youtube_upload.upload_rendered(
            output, privacy='private', client_secret=args.client_secret, token_file=token_file,
            interactive=False, on_progress=on_progress)

    client = WorkerClient(token, url, identifier, lease)
    stop = threading.Event()
    if install_signals:
        install_stop_signals(stop)
    log(f'워커 시작: {identifier} {url} 대기 {poll:.0f}초, 임대 {lease}초, 진행 보고 {interval:.0f}초')
    run_loop(client, root, args, poll, interval, once=args.once, upload=upload, credentials_ready=ready,
             stop=stop, log=log)


def xml_escape(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def agent_path(which=shutil.which):
    dirs = []
    for name in ('uv', 'ffmpeg', 'ffprobe', 'node'):
        found = which(name)
        if found:
            dirs.append(str(Path(found).parent))
    for extra in ('/opt/homebrew/bin', '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin'):
        dirs.append(extra)
    unique = []
    for item in dirs:
        if item not in unique:
            unique.append(item)
    return ':'.join(unique)


def render_launch_agent(repo, uv, log_path, path_env, template=None):
    text = Path(template or TEMPLATE).read_text(encoding='utf-8')
    values = {
        '__UV__': xml_escape(uv),
        '__REPO__': xml_escape(str(repo)),
        '__LOG__': xml_escape(str(log_path)),
        '__PATH__': xml_escape(path_env),
    }
    for key, value in values.items():
        if key not in text:
            raise RuntimeError(f'plist 템플릿에 {key}가 없습니다.')
        text = text.replace(key, value)
    plistlib.loads(text.encode('utf-8'))
    return text


def install_launch_agent(home=None, load=True, run=subprocess.run, which=shutil.which):
    uv = which('uv')
    if not uv:
        raise SystemExit('uv가 필요합니다. brew install uv')
    home = Path(home or Path.home())
    destination = home / 'Library' / 'LaunchAgents' / 'com.scripter.worker.plist'
    log_path = home / 'Library' / 'Logs' / 'scripter-worker.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = render_launch_agent(ROOT, uv, log_path, agent_path(which))
    destination.write_text(text, encoding='utf-8')
    print(f'plist: {destination}', flush=True)
    print(f'로그: {log_path}', flush=True)
    if not load:
        return destination
    if sys.platform != 'darwin':
        raise RuntimeError(f'launchd 등록은 macOS에서만 됩니다. plist는 써 두었습니다: {destination}')
    target = f'gui/{os.getuid()}'
    run(['launchctl', 'bootout', target, str(destination)], check=False)
    run(['launchctl', 'bootstrap', target, str(destination)], check=True)
    run(['launchctl', 'enable', f'{target}/com.scripter.worker'], check=False)
    run(['launchctl', 'kickstart', '-k', f'{target}/com.scripter.worker'], check=True)
    print('로그인 시 워커가 시작됩니다.', flush=True)
    return destination


def uninstall_launch_agent(home=None, run=subprocess.run):
    home = Path(home or Path.home())
    destination = home / 'Library' / 'LaunchAgents' / 'com.scripter.worker.plist'
    if sys.platform == 'darwin':
        run(['launchctl', 'bootout', f'gui/{os.getuid()}', str(destination)], check=False)
    if destination.exists():
        destination.unlink()
        print(f'제거했습니다: {destination}', flush=True)
    else:
        print(f'plist가 없습니다: {destination}', flush=True)


def validate_worker_args(args, parser):
    if args.once and not args.worker:
        parser.error('--once는 --worker와 함께 쓰세요.')
    if not args.worker:
        return
    if args.audio or args.input:
        parser.error('--worker는 입력 파일을 받지 않습니다. 큐에서 받습니다.')
    blocked = []
    for flag, active in (
        ('--upload', args.upload), ('--youtube-auth', args.youtube_auth),
        ('--no-video', args.no_video), ('--resume', args.resume),
        ('--thumbnail', args.thumbnail), ('--chapters', args.chapters),
        ('--transcript', args.transcript is not None),
    ):
        if active:
            blocked.append(flag)
    if blocked:
        parser.error('--worker는 ' + ', '.join(blocked) + '와 함께 쓸 수 없습니다.')
    if args.start or args.end is not None or args.duration is not None:
        parser.error('--worker는 녹음 전체를 처리합니다. --start, --end, --duration은 쓸 수 없습니다.')
    if args.privacy != 'private':
        parser.error('워커가 올리는 영상은 private만 됩니다. 확인 후 유튜브에서 공개하세요.')
    if not 30 <= args.lease_seconds <= 86400:
        parser.error('--lease-seconds는 30 이상 86400 이하여야 합니다.')
    poll = args.poll_seconds if args.poll_seconds is not None else 60
    if poll <= 0:
        parser.error('--poll-seconds는 양수여야 합니다.')
    if not os.environ.get('WORKER_TOKEN', '').strip():
        parser.error('워커에는 WORKER_TOKEN이 필요합니다. .env를 확인하세요.')
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        parser.error('워커에는 OPENAI_API_KEY가 필요합니다. 교정과 유튜브 자료를 만듭니다.')


def resolve_upload_target(args, parser):
    if args.output is not None:
        output = args.output
    else:
        try:
            audio = args.input or args.audio or None
            if audio is None:
                from scripter import default_audio
                audio = default_audio()
        except ValueError as error:
            parser.error(str(error))
        output = Path('files') / (Path(audio).stem + '.mp4')
    if output.suffix.lower() != '.mp4':
        parser.error('업로드 대상은 .mp4 경로여야 합니다.')
    if not output.is_file():
        parser.error(f'업로드할 영상이 없습니다: {output}')
    meta = output.with_suffix('.youtube.json')
    if not meta.is_file():
        parser.error(f'유튜브 제목·설명이 없습니다: {meta} (--youtube를 먼저 실행하세요)')
    return output
