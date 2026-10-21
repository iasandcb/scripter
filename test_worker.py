import json
import os
import plistlib
import stat
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import worker
from worker import (
    AuthError, LeaseLost, WorkerClient, WorkerError, audio_extension, heartbeat_interval,
    install_launch_agent, interpret_progress, keep_awake, pipeline_command, process_job,
    render_launch_agent, run_loop, safe_component, terminate_group, uninstall_launch_agent,
)
from scripter import main


class API(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        self._handle('GET')

    def do_POST(self):
        self._handle('POST')

    def _handle(self, method):
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else b''
        self.server.calls.append({
            'method': method, 'path': self.path,
            'authorization': self.headers.get('Authorization'),
            'claim': self.headers.get('X-Claim-Token'),
            'body': body,
        })
        status, payload, extra = self.server.route(method, self.path, body, self.headers)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), API)
        self.httpd.calls = []
        self.httpd.route = lambda *args: (500, {'detail': 'unset'}, {})
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=3)
        self.httpd.server_close()

    def client(self):
        return WorkerClient('secret-token', self.base, 'scripter-test', 3600, sleep=lambda seconds: None, retries=4)

    def test_claim_download_progress_and_complete(self):
        def route(method, path, body, headers):
            if path == '/api/worker/jobs?order=oldest&status=queued':
                return 200, {'jobs': [{'id': 'job-1'}]}, {}
            if path == '/api/worker/jobs/claim':
                posted = json.loads(body)
                self.assertEqual(posted, {'worker_id': 'scripter-test', 'lease_seconds': 3600})
                self.assertEqual(headers.get('Authorization'), 'Bearer secret-token')
                self.assertIsNone(headers.get('X-Claim-Token'))
                return 200, {'job': {
                    'id': 'job/1', 'claim_token': 'claim-1', 'title': '강의',
                    'recording_id': 'rec', 'mime': 'audio/mp4', 'duration_sec': 3,
                }}, {}
            if path == '/api/worker/jobs/job%2F1/audio':
                self.assertEqual(headers.get('X-Claim-Token'), 'claim-1')
                self.assertEqual(headers.get('Authorization'), 'Bearer secret-token')
                return 200, b'AUDIO', {'Content-Type': 'audio/mp4'}
            if path.endswith('/progress'):
                posted = json.loads(body)
                self.assertEqual(posted['claim_token'], 'claim-1')
                self.assertEqual(posted['lease_seconds'], 3600)
                self.assertEqual(posted['percent'], 40)
                return 200, {'ok': True}, {}
            if path.endswith('/complete'):
                posted = json.loads(body)
                self.assertEqual(posted['youtube_video_id'], 'abc')
                self.assertEqual(posted['youtube_url'], 'https://www.youtube.com/watch?v=abc')
                self.assertEqual(posted['transcript'], '교정된 말')
                return 200, {'ok': True}, {}
            if path.endswith('/fail'):
                return 200, {'ok': True}, {}
            return 404, {'detail': path}, {}

        self.httpd.route = route
        client = self.client()
        self.assertEqual(client.list_jobs(status='queued')[0]['id'], 'job-1')
        job = client.claim()
        self.assertEqual(job['id'], 'job/1')
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / 'audio.m4a'
            client.download_audio(job, audio)
            self.assertEqual(audio.read_bytes(), b'AUDIO')
        client.progress(job, message='렌더링', percent=40)
        client.complete(job, 'abc', 'https://www.youtube.com/watch?v=abc', '교정된 말')

    def test_empty_claim_auth_error_and_retry(self):
        hits = {'claim': 0}

        def route(method, path, body, headers):
            if path == '/api/worker/jobs/claim':
                hits['claim'] += 1
                if hits['claim'] == 1:
                    return 503, {'detail': 'busy'}, {}
                return 200, {'job': None}, {}
            return 404, {'detail': 'no'}, {}

        self.httpd.route = route
        self.assertIsNone(self.client().claim())
        self.assertEqual(hits['claim'], 2)

        def unauthorized(method, path, body, headers):
            return 401, {'detail': 'Not authenticated'}, {}

        self.httpd.route = unauthorized
        with self.assertRaises(AuthError):
            self.client().claim()

    def test_fail_posts_the_error(self):
        def route(method, path, body, headers):
            posted = json.loads(body)
            self.assertEqual(posted['error'], 'pipeline broke')
            self.assertEqual(headers.get('X-Claim-Token'), 'claim-1')
            return 200, {'ok': True}, {}

        self.httpd.route = route
        self.client().fail({'id': 'job-9', 'claim_token': 'claim-1'}, 'pipeline broke')


class PipelineCommandTests(unittest.TestCase):
    def test_worker_pipeline_uses_glossary_correct_and_youtube(self):
        import argparse
        options = argparse.Namespace(
            engine='auto', language='ko', youtube_model='gpt-6-luna',
            youtube_image_model='gpt-image-2.5-flare', correct_model='gpt-6-luna',
            correct_threshold=0.1, correct_force=False, model=None, glossary=None,
            spoken_math=None, spoken_math_url='', width=1920, height=1080, fps=30,
            font=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'), font_size=80,
            margin=120, lines=7)
        command = pipeline_command(Path('in.m4a'), Path('out.mp4'), options)
        self.assertIn('--correct', command)
        self.assertIn('--youtube', command)
        self.assertIn('--allow-missing-thumbnail', command)
        self.assertIn('--engine', command)
        self.assertNotIn('--glossary', command)
        options.glossary = ''
        self.assertIn('--glossary', pipeline_command(Path('in.m4a'), Path('out.mp4'), options))

    def test_progress_mapping_and_names(self):
        self.assertEqual(interpret_progress('인식: 0.0 / 10.0초 (50.0%)', '시작', 0), ('음성 인식 중', 20))
        self.assertEqual(interpret_progress('렌더링: 5.0 / 10.0초 (50.0%)', '음성 인식 중', 20)[0], '영상 렌더링 중')
        self.assertEqual(interpret_progress('교정 중...', '음성 인식 중', 20)[0], '전사문 교정 중')
        self.assertEqual(audio_extension('audio/mpeg; codecs=mp3'), '.mp3')
        self.assertEqual(audio_extension(None, 'attachment; filename="talk.wav"'), '.wav')
        self.assertEqual(safe_component('job/1'), 'job_1')
        self.assertEqual(heartbeat_interval(120, 3600), 120)
        self.assertEqual(heartbeat_interval(120, 30), 10)


def write_outputs(output, transcript='교정된 전사'):
    output.write_bytes(b'not a real mp4')
    output.with_suffix('.txt').write_text(transcript, encoding='utf-8')
    output.with_suffix('.youtube.json').write_text('{"title":"제목","description":"설명"}\n', encoding='utf-8')


class JobTests(unittest.TestCase):
    def job(self, token='claim-1'):
        return {'id': 'job-1', 'claim_token': token, 'title': '강의', 'recording_id': 'rec',
                'mime': 'audio/mp4', 'duration_sec': 12}

    def test_complete_retries_without_a_second_upload(self):
        uploads = []
        completes = {'n': 0}

        class Client:
            def download_audio(self, job, destination):
                destination.write_bytes(b'audio')

            def progress(self, job, message=None, percent=None):
                return None

            def complete(self, job, video_id, url, transcript):
                completes['n'] += 1
                self.last = (job['claim_token'], video_id, url, transcript)
                if completes['n'] == 1:
                    raise WorkerError('complete dropped')

            def fail(self, job, error):
                raise AssertionError(error)

        def pipeline(audio, output, cancel, status, options, log):
            write_outputs(output)

        def upload(output, on_progress=None):
            uploads.append(output)
            return {'youtube_video_id': 'abc', 'youtube_url': 'https://www.youtube.com/watch?v=abc', 'thumbnail': False}

        client = Client()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(worker.CompletePending):
                process_job(client, self.job(), folder, None, pipeline, upload, lambda message: None,
                            lambda seconds: None, 30, 1)
            state_path = Path(folder) / 'job-1' / 'state.json'
            state = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual(state['phase'], 'uploaded')
            self.assertEqual(state['youtube_video_id'], 'abc')
            self.assertEqual(state['transcript'], '교정된 전사')
            self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)
            process_job(client, self.job(token='claim-2'), folder, None, pipeline, upload, lambda message: None,
                        lambda seconds: None, 30, 1)
            self.assertEqual(len(uploads), 1)
            self.assertEqual(client.last[0], 'claim-2')
            self.assertEqual(client.last[3], '교정된 전사')
            self.assertEqual(json.loads(state_path.read_text(encoding='utf-8'))['phase'], 'completed')

    def test_lease_loss_does_not_upload_or_fail(self):
        uploads = []
        failed = []
        progressed = {'n': 0}

        class Client:
            def claim(self):
                return {'id': 'job-1', 'claim_token': 'claim', 'title': '강의', 'mime': 'audio/mp4'}

            def download_audio(self, job, destination):
                destination.write_bytes(b'audio')

            def progress(self, job, message=None, percent=None):
                progressed['n'] += 1
                if progressed['n'] > 2:
                    raise LeaseLost('lease expired', status=409)

            def complete(self, *args):
                raise AssertionError('complete')

            def fail(self, job, error):
                failed.append(error)

        def pipeline(audio, output, cancel, status, options, log):
            if not cancel.wait(2):
                raise AssertionError('expected the lease heartbeat to cancel the pipeline')

        def upload(output, on_progress=None):
            uploads.append(output)

        @contextmanager
        def awake():
            awake.called = True
            yield

        awake.called = False
        client = Client()
        with tempfile.TemporaryDirectory() as folder, patch('worker.keep_awake', awake):
            with self.assertRaises(SystemExit) as caught:
                run_loop(client, folder, None, 60, 0.01, once=True, pipeline=pipeline, upload=upload,
                         sleep=lambda seconds: None, log=lambda message: None, complete_attempts=1)
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(uploads, [])
        self.assertEqual(failed, [])
        self.assertTrue(awake.called)

    def test_idle_poll_does_not_keep_the_machine_awake(self):
        class Client:
            def claim(self):
                return None

        @contextmanager
        def awake():
            raise AssertionError('caffeinate')

        with patch('worker.keep_awake', awake):
            run_loop(Client(), Path('/tmp'), None, 60, 30, once=True, sleep=lambda seconds: None,
                     log=lambda message: None)

    def test_missing_youtube_auth_does_not_claim(self):
        class Client:
            def claim(self):
                raise AssertionError('claimed')

        with self.assertRaises(SystemExit):
            run_loop(Client(), Path('/tmp'), None, 60, 30, once=True, credentials_ready=lambda: False,
                     sleep=lambda seconds: None, log=lambda message: None)

    def test_failed_job_is_reported_and_pipeline_is_not_repeated(self):
        pipelines = []

        class Client:
            def __init__(self):
                self.failed = []
                self.jobs = [ {'id': 'job-1', 'claim_token': 't', 'title': '강의', 'mime': 'audio/mp4'} ]

            def claim(self):
                return self.jobs.pop(0) if self.jobs else None

            def download_audio(self, job, destination):
                destination.write_bytes(b'audio')

            def progress(self, job, message=None, percent=None):
                return None

            def complete(self, *args):
                raise AssertionError('complete')

            def fail(self, job, error):
                self.failed.append(error)

        def pipeline(audio, output, cancel, status, options, log):
            pipelines.append(1)
            write_outputs(output)

        def upload(output, on_progress=None):
            raise RuntimeError('upload broke')

        client = Client()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit):
                run_loop(client, folder, None, 15, 30, once=True, pipeline=pipeline, upload=upload,
                         sleep=lambda seconds: None, log=lambda message: None, complete_attempts=1)
            self.assertEqual(pipelines, [1])
            self.assertIn('upload broke', client.failed[0])
            with self.assertRaises(RuntimeError):
                process_job(client, {'id': 'job-1', 'claim_token': 't2', 'mime': 'audio/mp4'}, folder, None,
                            pipeline, upload, lambda message: None, lambda seconds: None, 30, 1)
            self.assertEqual(pipelines, [1])


class AwakeTests(unittest.TestCase):
    def test_caffeinate_starts_and_stops_on_darwin(self):
        with patch('worker.sys.platform', 'darwin'), \
             patch('worker.shutil.which', return_value='/usr/bin/caffeinate'), \
             patch('worker.subprocess.Popen') as popen:
            proc = popen.return_value
            proc.poll.return_value = None
            proc.wait.return_value = 0
            with keep_awake():
                popen.assert_called_once()
                self.assertEqual(popen.call_args.args[0], ['caffeinate', '-i'])
            proc.terminate.assert_called_once()

    def test_terminate_group_stops_a_process(self):
        import subprocess
        proc = subprocess.Popen(['sleep', '30'], start_new_session=True)
        terminate_group(proc)
        self.assertIsNotNone(proc.poll())


class LaunchAgentTests(unittest.TestCase):
    def test_template_and_installer(self):
        text = render_launch_agent('/Users/me/scripter & notes', '/opt/homebrew/bin/uv',
                                   '/Users/me/Library/Logs/scripter-worker.log', '/opt/homebrew/bin:/usr/bin')
        self.assertIn('/Users/me/scripter &amp; notes', text)
        loaded = plistlib.loads(text.encode('utf-8'))
        self.assertEqual(loaded['Label'], 'com.scripter.worker')
        self.assertEqual(loaded['ProgramArguments'], [
            '/opt/homebrew/bin/uv', 'run', '--directory', '/Users/me/scripter & notes', 'scripter.py', '--worker'])
        self.assertTrue(loaded['RunAtLoad'])
        self.assertTrue(loaded['KeepAlive'])
        self.assertEqual(loaded['ThrottleInterval'], 30)
        self.assertEqual(loaded['StandardOutPath'], loaded['StandardErrorPath'])
        self.assertEqual(loaded['WorkingDirectory'], '/Users/me/scripter & notes')

        def which(name):
            return '/opt/homebrew/bin/uv' if name == 'uv' else None

        with tempfile.TemporaryDirectory() as folder:
            destination = install_launch_agent(home=folder, load=False, which=which)
            plist = plistlib.loads(destination.read_bytes())
            self.assertIn('--worker', plist['ProgramArguments'])
            self.assertTrue(str(destination).endswith('Library/LaunchAgents/com.scripter.worker.plist'))
            self.assertIn('scripter-worker.log', plist['StandardOutPath'])

            def refuse(*args, **kwargs):
                raise AssertionError('launchctl should not run off macOS or when load is false')

            uninstall_launch_agent(home=folder, run=refuse)
            self.assertFalse(destination.exists())


class CliTests(unittest.TestCase):
    def test_worker_flags(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['--once'])
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['--worker', 'talk.m4a'])
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['--worker', '--no-video'])
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['--worker', '--privacy', 'public'])
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            main(['--worker', '--lease-seconds', '10'])
        with patch('worker.serve') as serve, \
             patch.dict(os.environ, {'WORKER_TOKEN': 'token', 'OPENAI_API_KEY': 'key'}):
            main(['--worker', '--once'])
        serve.assert_called_once()
        self.assertTrue(serve.call_args.args[0].once)

    def test_allow_missing_thumbnail_continues(self):
        words = [{'text': '안녕하세요', 'start': 1, 'end': 2}]

        def boom(text, output, *args, **kwargs):
            output.with_suffix('.youtube.json').write_text('{"title":"제목"}\n', encoding='utf-8')
            raise RuntimeError('썸네일 실패. 제목·설명은 저장되었습니다.')

        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / 'talk.m4a'
            output = Path(folder) / 'talk.mp4'
            transcript = output.with_suffix('.words.json')
            transcript.write_text(json.dumps(words), encoding='utf-8')
            argv = ['-i', str(audio), '-o', str(output), '--transcript', str(transcript), '--youtube',
                    '--allow-missing-thumbnail']
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                 patch('scripter.shutil.which', return_value='tool'), \
                 patch('scripter.fetch_spoken_math', return_value=None), \
                 patch('scripter.Path.is_file', lambda path: path.suffix in ('.m4a', '.ttc', '.json') or path.exists()), \
                 patch('scripter.probe', return_value=10), patch('scripter.mlx_available', return_value=False), \
                 patch('scripter.render'), patch('scripter.generate_youtube', side_effect=boom):
                main(argv)
            self.assertTrue(output.with_suffix('.youtube.json').is_file())
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                 patch('scripter.shutil.which', return_value='tool'), \
                 patch('scripter.fetch_spoken_math', return_value=None), \
                 patch('scripter.Path.is_file', lambda path: path.suffix in ('.m4a', '.ttc', '.json') or path.exists()), \
                 patch('scripter.probe', return_value=10), patch('scripter.mlx_available', return_value=False), \
                 patch('scripter.render'), patch('scripter.generate_youtube', side_effect=boom), \
                 self.assertRaises(RuntimeError):
                main(['-i', str(audio), '-o', str(output), '--transcript', str(transcript), '--youtube'])


if __name__ == '__main__':
    unittest.main()
