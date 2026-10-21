import base64
import hashlib
import json
import os
import stat
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from net import atomic_write, request
from youtube_upload import (
    Credentials, YouTubeClient, authorization_url, description_with_chapters,
    load_client, parse_callback, pkce_pair, video_body, wait_for_callback, _Callback,
)


def client_json(folder):
    path = Path(folder) / 'client_secret.json'
    path.write_text(json.dumps({'installed': {
        'client_id': 'client',
        'client_secret': 'super-secret',
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'redirect_uris': ['http://localhost'],
    }}), encoding='utf-8')
    return path


def token_json(folder, expires_at):
    path = Path(folder) / 'token.json'
    path.write_text(json.dumps({
        'refresh_token': 'refresh',
        'access_token': 'access',
        'expires_at': expires_at,
        'client_id': 'client',
    }), encoding='utf-8')
    os.chmod(path, 0o600)
    return path


class OAuthTests(unittest.TestCase):
    def test_authorization_url_and_pkce(self):
        verifier, challenge = pkce_pair()
        digest = hashlib.sha256(verifier.encode('ascii')).digest()
        self.assertEqual(challenge, base64.urlsafe_b64encode(digest).decode('ascii').rstrip('='))
        self.assertNotIn('=', verifier)
        with tempfile.TemporaryDirectory() as folder:
            config = load_client(client_json(folder))
        url = authorization_url(config, 'http://127.0.0.1:9/', 'state-1', challenge)
        self.assertIn('access_type=offline', url)
        self.assertIn('prompt=consent', url)
        self.assertIn('code_challenge_method=S256', url)
        self.assertIn('youtube.upload', url)
        self.assertIn('youtube.force-ssl', url)
        self.assertIn('state-1', url)

    def test_callback_rejects_a_mismatched_state(self):
        with self.assertRaises(RuntimeError):
            parse_callback({'code': ['abc'], 'state': ['nope']}, 'expected')
        with self.assertRaises(RuntimeError):
            parse_callback({'error': ['access_denied'], 'state': ['expected']}, 'expected')
        self.assertEqual(parse_callback({'code': ['abc'], 'state': ['expected']}, 'expected'), 'abc')

    def test_loopback_records_the_code(self):
        server = HTTPServer(('127.0.0.1', 0), _Callback)
        server.result = None
        box = {}

        def serve():
            box['query'] = wait_for_callback(server, timeout=5)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/favicon.ico', timeout=2) as favicon:
                self.assertEqual(favicon.status, 200)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/?code=from-browser&state=abc', timeout=2) as response:
                self.assertIn('인증이 끝났습니다', response.read().decode('utf-8'))
            thread.join(timeout=5)
        finally:
            server.server_close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(parse_callback(box['query'], 'abc'), 'from-browser')

    def test_exchange_stores_refresh_token_mode_600(self):
        def transport(method, url, data=None, headers=None, timeout=60):
            form = (data or b'').decode('utf-8')
            self.assertIn('grant_type=authorization_code', form)
            self.assertIn('code_verifier=verifier', form)
            self.assertNotIn('super-secret-in-body-only', form)  # secret is in the form, checked below
            self.assertIn('client_secret=super-secret', form)
            return 200, {}, json.dumps({
                'access_token': 'ya', 'refresh_token': 'yr', 'expires_in': 3600, 'token_type': 'Bearer',
            }).encode()

        with tempfile.TemporaryDirectory() as folder:
            secret = client_json(folder)
            token = Path(folder) / 'saved.json'
            credentials = Credentials(secret, token, transport=transport)
            credentials.exchange('code', 'http://127.0.0.1:9/', 'verifier')
            mode = stat.S_IMODE(token.stat().st_mode)
            self.assertEqual(mode, 0o600)
            saved = json.loads(token.read_text(encoding='utf-8'))
            self.assertEqual(saved['refresh_token'], 'yr')
            self.assertNotIn('client_secret', saved)
            self.assertNotIn('super-secret', token.read_text(encoding='utf-8'))

    def test_fresh_access_token_skips_the_network(self):
        def transport(*args, **kwargs):
            raise AssertionError('network')

        with tempfile.TemporaryDirectory() as folder:
            credentials = Credentials(client_json(folder), token_json(folder, time.time() + 600), transport=transport)
            self.assertEqual(credentials.authorization(), 'Bearer access')

    def test_expired_token_refreshes(self):
        def transport(method, url, data=None, headers=None, timeout=60):
            self.assertIn('grant_type=refresh_token', (data or b'').decode())
            self.assertIn('refresh_token=refresh', (data or b'').decode())
            return 200, {}, json.dumps({
                'access_token': 'new-access', 'expires_in': 100, 'token_type': 'Bearer',
            }).encode()

        with tempfile.TemporaryDirectory() as folder:
            token = token_json(folder, time.time() - 10)
            credentials = Credentials(client_json(folder), token, transport=transport)
            self.assertEqual(credentials.authorization(), 'Bearer new-access')
            self.assertEqual(json.loads(token.read_text())['refresh_token'], 'refresh')


class MetadataTests(unittest.TestCase):
    def test_private_korean_body_and_chapters(self):
        metadata = {
            'title': '강의 제목', 'description': '설명입니다.', 'thumbnail_prompt': '그림',
            'chapters': [{'start': '00:00', 'title': '시작'}, {'start': '01:00', 'title': '본론'}],
        }
        body = video_body(metadata)
        self.assertEqual(body['status']['privacyStatus'], 'private')
        self.assertIs(body['status']['selfDeclaredMadeForKids'], False)
        self.assertEqual(body['snippet']['defaultLanguage'], 'ko')
        self.assertEqual(body['snippet']['defaultAudioLanguage'], 'ko')
        self.assertEqual(body['snippet']['categoryId'], '27')
        self.assertEqual(body['snippet']['title'], '강의 제목')
        self.assertIn('설명입니다.', body['snippet']['description'])
        self.assertIn('00:00 시작', body['snippet']['description'])
        self.assertIn('01:00 본론', body['snippet']['description'])
        self.assertNotIn('그림', body['snippet']['description'])

    def test_chapters_are_dropped_to_keep_the_description_limit(self):
        metadata = {'description': '가' * 4990, 'chapters': [
            {'start': '00:00', 'title': '하나'}, {'start': '00:10', 'title': '둘둘둘둘'}]}
        text = description_with_chapters(metadata)
        self.assertLessEqual(len(text), 5000)
        self.assertIn('00:00 하나', text)
        self.assertNotIn('둘둘둘둘', text)


class _Hold(BaseHTTPRequestHandler):
    def do_PUT(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length:
            self.rfile.read(length)
        self.send_response(308)
        self.send_header('Range', 'bytes=0-3')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, fmt, *args):
        return


class NetTests(unittest.TestCase):
    def test_308_is_not_followed(self):
        server = HTTPServer(('127.0.0.1', 0), _Hold)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_address[1]}/upload'
            status, headers, body = request('PUT', url, data=b'abcd', headers={'Content-Range': 'bytes 0-3/8'})
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertEqual(status, 308)
        self.assertIn('bytes=0-3', headers.get('Range'))

    def test_atomic_write_refuses_a_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'real'
            link = Path(folder) / 'link'
            target.write_text('x', encoding='utf-8')
            link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                atomic_write(link, 'secret')
            self.assertEqual(target.read_text(encoding='utf-8'), 'x')


class UploadTests(unittest.TestCase):
    def test_resumable_upload_and_thumbnail_failure(self):
        calls = []

        def transport(method, url, data=None, headers=None, timeout=60):
            headers = dict(headers or {})
            calls.append((method, url, data, headers))
            content_range = headers.get('Content-Range', '')
            if method == 'POST' and 'uploadType=resumable' in url:
                posted = json.loads(data.decode('utf-8'))
                self.assertEqual(posted['status']['privacyStatus'], 'private')
                self.assertEqual(posted['snippet']['defaultLanguage'], 'ko')
                self.assertIn('00:00 시작', posted['snippet']['description'])
                self.assertEqual(headers['Authorization'], 'Bearer access')
                self.assertEqual(headers['X-Upload-Content-Length'], '20')
                return 200, {'Location': 'https://upload.example/session'}, b''
            if method == 'PUT' and content_range.startswith('bytes */'):
                return 308, {'Range': 'bytes=0-7'}, b''
            if method == 'PUT' and content_range.startswith('bytes 0-'):
                raise TimeoutError('connection reset')
            if method == 'PUT' and content_range.startswith('bytes 8-'):
                return 308, {'Range': 'bytes=0-15'}, b''
            if method == 'PUT' and content_range.startswith('bytes 16-'):
                self.assertEqual(content_range, 'bytes 16-19/20')
                return 200, {'Content-Type': 'application/json'}, b'{"id":"vid123"}'
            if method == 'POST' and 'thumbnails' in url:
                self.assertIn('videoId=vid123', url)
                return 403, {}, b'{"error":{"message":"custom thumbnail permission"}}'
            raise AssertionError((method, url, content_range))

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / 'video.mp4'
            video.write_bytes(b'0123456789abcdefghij')
            video.with_suffix('.youtube.json').write_text(json.dumps({
                'title': '강의 제목', 'description': '설명입니다.',
                'chapters': [{'start': '00:00', 'title': '시작'}, {'start': '01:00', 'title': '본론'}],
            }, ensure_ascii=False), encoding='utf-8')
            video.with_suffix('.thumbnail.png').write_bytes(b'\x89PNG')
            credentials = Credentials(client_json(folder), token_json(folder, time.time() + 600), transport=transport)
            client = YouTubeClient(credentials, transport=transport, chunk_size=8, sleep=lambda seconds: None)
            result = client.upload(video, json.loads(video.with_suffix('.youtube.json').read_text(encoding='utf-8')))
            self.assertEqual(result['youtube_video_id'], 'vid123')
            self.assertEqual(result['youtube_url'], 'https://www.youtube.com/watch?v=vid123')
            self.assertIs(result['thumbnail'], False)
            puts = [call for call in calls if call[0] == 'PUT']
            result = client.upload(video, json.loads(video.with_suffix('.youtube.json').read_text(encoding='utf-8')))
            puts_after = [call for call in calls if call[0] == 'PUT']
            self.assertEqual(puts, puts_after)
            self.assertEqual(result['youtube_video_id'], 'vid123')
            marker = json.loads(video.with_suffix('.youtube-upload.json').read_text(encoding='utf-8'))
            self.assertEqual(marker['youtube_video_id'], 'vid123')
            self.assertEqual(stat.S_IMODE(video.with_suffix('.youtube-upload.json').stat().st_mode), 0o600)


if __name__ == '__main__':
    unittest.main()
