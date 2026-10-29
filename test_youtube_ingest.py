import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from net import request

from youtube_ingest import (
    AUDIO_FORMAT, ENQUEUE, IngestError, audio_filename, build_multipart, command_failed,
    download_local, encode_multipart, extract_ids, ffmpeg_remux_args, format_id_lines,
    ingest_url, main, parse_args, post_audio, quote_filename, read_token, remote_shell,
    run_streaming, validate_ssh_host, validate_youtube_url, ytdlp_args,
)


URL = 'https://www.youtube.com/watch?v=abcdefghijk'
TOKEN = 'super-secret-ingest-token'


def tools(name):
    return '/usr/bin/' + name


def completed(args, stdout='', stderr='', code=0):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


def multipart_parts(body, content_type):
    boundary = content_type.split('boundary=', 1)[1].strip().strip('"')
    delimiter = b'--' + boundary.encode('ascii')
    parts = []
    for piece in body.split(delimiter)[1:]:
        if piece.startswith(b'--'):
            break
        if piece.startswith(b'\r\n'):
            piece = piece[2:]
        if piece.endswith(b'\r\n'):
            piece = piece[:-2]
        header_blob, sep, data = piece.partition(b'\r\n\r\n')
        if not sep:
            raise AssertionError('multipart part has no header break')
        parts.append((header_blob, data))
    return parts


def filename_as_server_sees_it(header_blob):
    """Match python-multipart: latin-1 header bytes, quoted filename, then UTF-8.

    filename* is ignored there (RFC 7578). Decoding the header as latin-1 and
    encoding that string back to latin-1 restores the original UTF-8 bytes.
    """
    text = header_blob.decode('latin-1')
    marker = 'filename="'
    start = text.find(marker)
    if start < 0:
        return None
    chars = []
    index = start + len(marker)
    while index < len(text):
        char = text[index]
        if char == '\\':
            chars.append(text[index + 1])
            index += 2
            continue
        if char == '"':
            break
        chars.append(char)
        index += 1
    return ''.join(chars).encode('latin-1').decode('utf-8')


class FilenameTests(unittest.TestCase):
    def test_comma_and_quotes_stay_a_single_path_component(self):
        name = audio_filename('Say "hi", ok')
        self.assertEqual(name, 'Say "hi", ok.m4a')
        self.assertNotIn('/', name)
        self.assertNotIn('\\', name)

    def test_korean_comma_title(self):
        self.assertEqual(audio_filename('강의, 1부'), '강의, 1부.m4a')

    def test_path_traversal_cannot_leave_the_directory(self):
        name = audio_filename('../../secret, 1')
        self.assertEqual(name, 'secret, 1.m4a')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / name
            self.assertEqual(path.parent, Path(folder))
            self.assertFalse(path.is_absolute() and path.parent == Path(folder).parent)

    def test_newlines_do_not_become_headers(self):
        name = audio_filename('bad\r\nX-Evil: 1')
        self.assertNotIn('\n', name)
        self.assertNotIn('\r', name)
        quoted = quote_filename('bad\r\nX-Evil.m4a')
        self.assertNotIn('\n', quoted)
        self.assertNotIn('\r', quoted)

    def test_empty_title(self):
        self.assertEqual(audio_filename('   '), 'audio.m4a')
        self.assertEqual(audio_filename('..'), 'audio.m4a')


class MultipartTests(unittest.TestCase):
    def test_comma_title_is_one_file_part(self):
        body, content_type = encode_multipart(
            'Hello, world.m4a', b'--false\r\ncomma, inside', (('enqueue', ENQUEUE),))
        parts = multipart_parts(body, content_type)
        self.assertEqual(len(parts), 2)
        self.assertIn(b'name="enqueue"', parts[0][0])
        self.assertEqual(parts[0][1], b'false')
        self.assertIn(b'name="file"', parts[1][0])
        self.assertEqual(parts[1][1], b'--false\r\ncomma, inside')
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), 'Hello, world.m4a')
        self.assertNotIn(b'filename*', parts[1][0])

    def test_korean_title_roundtrips_through_latin1_headers(self):
        title = '강의, 1부 "핵심".m4a'
        body, content_type = encode_multipart(title, b'audio', (('enqueue', 'false'),))
        parts = multipart_parts(body, content_type)
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), '강의, 1부 "핵심".m4a')
        header_lines = [line for line in parts[1][0].split(b'\r\n') if line]
        self.assertEqual(
            [line.split(b':', 1)[0] for line in header_lines],
            [b'Content-Disposition', b'Content-Type'],
        )

    def test_semicolon_stays_inside_the_quoted_filename(self):
        body, content_type = encode_multipart('a;b, c.m4a', b'x', (('enqueue', 'false'),))
        parts = multipart_parts(body, content_type)
        self.assertEqual(len(parts), 2)
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), 'a;b, c.m4a')

    def test_boundary_that_appears_in_the_file_is_not_used(self):
        content = b'prefix ----scripterFIXED suffix'

        def factory():
            factory.calls += 1
            if factory.calls == 1:
                return '----scripterFIXED'
            return '----scripterCLEAN1'

        factory.calls = 0
        body, content_type = encode_multipart(
            'Hello, world.m4a', content, (('enqueue', 'false'),), boundary_factory=factory)
        self.assertIn('boundary=----scripterCLEAN1', content_type)
        parts = multipart_parts(body, content_type)
        self.assertEqual(parts[1][1], content)

    def test_build_multipart_reads_the_filename_from_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Hello, world.m4a'
            path.write_bytes(b'm4a-bytes')
            body, content_type = build_multipart(path)
        parts = multipart_parts(body, content_type)
        self.assertEqual(parts[0][1], b'false')
        self.assertEqual(parts[1][1], b'm4a-bytes')
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), 'Hello, world.m4a')
        self.assertNotIn(TOKEN.encode(), body)


class IdTests(unittest.TestCase):
    def test_snake_camel_and_nested_objects(self):
        self.assertEqual(
            format_id_lines({'project_id': 'p1', 'conversation_id': 'c1'}),
            ['project_id: p1', 'conversation_id: c1'])
        self.assertEqual(
            format_id_lines({'projectId': 'p2', 'conversationId': 'c2'}),
            ['project_id: p2', 'conversation_id: c2'])
        nested = format_id_lines({
            'data': {
                'project': {'id': 'p3', 'name': '강의'},
                'conversation': {'id': 'c3', 'project_id': 'p3'},
            },
        })
        self.assertEqual(nested, ['project_id: p3', 'conversation_id: c3'])

    def test_root_id_is_the_conversation_when_it_differs_from_the_project(self):
        self.assertEqual(
            format_id_lines({'id': 'c4', 'project_id': 'p4'}),
            ['project_id: p4', 'conversation_id: c4'])
        self.assertEqual(format_id_lines({'id': 'only'}), ['conversation_id: only'])
        self.assertEqual(extract_ids({'id': 'same', 'project_id': 'same'})['conversation_ids'], [])

    def test_lists_and_bools(self):
        found = extract_ids({
            'project_id': True,
            'project': {'id': 'p5'},
            'conversation_ids': ['c5', 'c6', 'c5'],
        })
        self.assertEqual(found['project_ids'], ['p5'])
        self.assertEqual(found['conversation_ids'], ['c5', 'c6'])


class TokenTests(unittest.TestCase):
    def test_read_token_does_not_print_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ingest_token'
            path.write_text(TOKEN + '\n', encoding='utf-8')
            os.chmod(path, 0o600)
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(read_token(path), TOKEN)
            self.assertNotIn(TOKEN, buf.getvalue())

    def test_loose_permissions_warn_without_the_token(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ingest_token'
            path.write_text(TOKEN, encoding='utf-8')
            os.chmod(path, 0o644)
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(read_token(path), TOKEN)
            self.assertIn('chmod 600', buf.getvalue())
            self.assertNotIn(TOKEN, buf.getvalue())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0o044)

    def test_symlink_and_multiline_are_rejected_without_echoing_the_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            real = folder / 'real'
            real.write_text(TOKEN, encoding='utf-8')
            link = folder / 'link'
            link.symlink_to(real)
            with self.assertRaises(IngestError) as linked:
                read_token(link)
            self.assertNotIn(TOKEN, str(linked.exception))
            os.chmod(real, 0o600)
            real.write_text(TOKEN + '\nalso ' + TOKEN, encoding='utf-8')
            with self.assertRaises(IngestError) as lines:
                read_token(real)
            self.assertNotIn(TOKEN, str(lines.exception))


class CommandTests(unittest.TestCase):
    def test_ytdlp_downloads_native_m4a_without_reencoding(self):
        args = ytdlp_args(URL, 'files/%(title)s.%(ext)s', ['after_move:filepath'])
        self.assertIsInstance(args, list)
        self.assertEqual(args[0], 'yt-dlp')
        self.assertEqual(args[-1], URL)
        self.assertIn('--no-simulate', args)
        self.assertIn(AUDIO_FORMAT, args)
        self.assertNotIn('-x', args)
        self.assertNotIn('--extract-audio', args)
        self.assertNotIn('--audio-format', args)
        self.assertNotIn('ffmpeg', args)

    def test_remote_command_quotes_a_hostile_url(self):
        url = "https://www.youtube.com/watch?v=abc';touch${IFS}pwned"
        remote_dir = '/tmp/scripter-youtube-ingest-' + ('a' * 32)
        script = remote_shell(url, remote_dir)
        parts = shlex.split(script)
        self.assertIn(url, parts)
        self.assertNotIn('touch', parts)
        self.assertNotIn('-x', parts)
        self.assertIn('yt-dlp', parts)
        self.assertIn('/opt/homebrew/bin', script)
        self.assertTrue(script.startswith('export PATH='))

    def test_ffmpeg_remux_is_copy_only(self):
        args = ffmpeg_remux_args('Say "hi", ok.m4a', 'out.m4a')
        self.assertEqual(args[args.index('-c') + 1], 'copy')
        self.assertNotIn('aac', args)
        self.assertNotIn('libx264', args)
        self.assertEqual(args[0], 'ffmpeg')

    def test_bot_check_suggests_ssh_only_when_this_machine_failed(self):
        local = command_failed('yt-dlp', 1, 'ERROR: Sign in to confirm you’re not a bot')
        self.assertIn('--via-ssh', str(local))
        remote = command_failed('ssh', 1, 'ERROR: Sign in to confirm you\'re not a bot', via_ssh=True)
        self.assertNotIn('--via-ssh', str(remote))
        self.assertIn('원격 맥', str(remote))

    def test_urls_and_hosts(self):
        self.assertEqual(validate_youtube_url('https://youtu.be/abc'), 'https://youtu.be/abc')
        self.assertEqual(
            validate_youtube_url('https://www.youtube.com/watch?v=abc&t=1'),
            'https://www.youtube.com/watch?v=abc&t=1')
        with self.assertRaises(IngestError):
            validate_youtube_url('https://example.com/watch?v=abc')
        with self.assertRaises(IngestError):
            validate_youtube_url('file:///tmp/a.m4a')
        self.assertEqual(validate_ssh_host('me@macbook.local'), 'me@macbook.local')
        for host in ('-oProxyCommand=evil', 'host name', 'user@host;rm', ''):
            with self.assertRaises(IngestError):
                validate_ssh_host(host)


class _IngestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else b''
        self.server.seen = {
            'path': self.path,
            'authorization': self.headers.get('Authorization'),
            'content_type': self.headers.get('Content-Type'),
            'body': body,
        }
        payload = json.dumps({'project_id': 'proj-1', 'conversation_id': 'conv-1'}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        return


class IngestTests(unittest.TestCase):
    def test_local_download_posts_the_comma_filename(self):
        seen = {}

        def transport(method, url, data=None, headers=None, timeout=60):
            seen.update(method=method, url=url, data=data, headers=headers)
            self.assertEqual(headers['Authorization'], 'Bearer ' + TOKEN)
            self.assertNotIn(TOKEN.encode(), data)
            return 200, {}, json.dumps({
                'project': {'id': 'proj-9'},
                'conversation': {'id': 'conv-9'},
            }).encode()

        def run(args, **kwargs):
            self.assertIsInstance(args, list)
            self.assertEqual(args[0], 'yt-dlp')
            self.assertNotIn('-x', args)
            self.assertNotIn(TOKEN, ' '.join(args))
            template = args[args.index('-o') + 1]
            path = Path(template.replace('%(title)s', 'Hello, world').replace('%(ext)s', 'm4a'))
            path.write_bytes(b'native-m4a')
            return completed(args, stdout=str(path) + '\n')

        with tempfile.TemporaryDirectory() as folder:
            path, lines = ingest_url(
                URL, output_dir=folder, token=TOKEN, transport=transport, run=run, which=tools)
        self.assertEqual(path.name, 'Hello, world.m4a')
        self.assertEqual(lines, ['project_id: proj-9', 'conversation_id: conv-9'])
        self.assertEqual(seen['method'], 'POST')
        self.assertTrue(seen['url'].endswith('/api/ingest/audio'))
        parts = multipart_parts(seen['data'], seen['headers']['Content-Type'])
        self.assertEqual(parts[0][1], b'false')
        self.assertEqual(parts[1][1], b'native-m4a')
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), 'Hello, world.m4a')
        self.assertNotIn(TOKEN, '\n'.join(lines))

    def test_ssh_downloads_remotely_and_remuxes_before_upload(self):
        seen = {}
        calls = []

        def transport(method, url, data=None, headers=None, timeout=60):
            seen['data'] = data
            seen['headers'] = headers
            seen['url'] = url
            return 200, {}, b'{"project_id":"p","conversation_id":"c"}'

        def run(args, **kwargs):
            calls.append(list(args))
            self.assertNotIn(TOKEN, ' '.join(str(part) for part in args))
            if args[0] == 'ssh' and 'rm -rf' in args[-1]:
                return completed(args)
            if args[0] == 'ssh':
                remote_dir = '/tmp/scripter-youtube-ingest-' + args[-1].split('/tmp/scripter-youtube-ingest-', 1)[1][:32]
                stdout = f'Say "hi", ok\n{remote_dir}/audio.m4a\n'
                self.assertNotIn('-x', shlex.split(args[-1]))
                self.assertNotIn('ffmpeg', shlex.split(args[-1]))
                return completed(args, stdout=stdout)
            if args[0] == 'scp':
                Path(args[-1]).write_bytes(b'from-remote')
                return completed(args)
            if args[0] == 'ffmpeg':
                self.assertEqual(args[args.index('-c') + 1], 'copy')
                Path(args[-1]).write_bytes(b'remuxed-m4a')
                return completed(args)
            raise AssertionError(args)

        def which(name):
            if name == 'yt-dlp':
                return None
            return '/usr/bin/' + name

        with tempfile.TemporaryDirectory() as folder:
            path, lines = ingest_url(
                URL, via_ssh='me@macbook', output_dir=folder, token=TOKEN,
                transport=transport, run=run, which=which)
            self.assertEqual(path.parent, Path(folder))
            self.assertEqual(path.name, 'Say "hi", ok.m4a')
            self.assertEqual(path.read_bytes(), b'remuxed-m4a')
        self.assertEqual(lines, ['project_id: p', 'conversation_id: c'])
        parts = multipart_parts(seen['data'], seen['headers']['Content-Type'])
        self.assertEqual(parts[1][1], b'remuxed-m4a')
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), 'Say "hi", ok.m4a')
        self.assertEqual(seen['headers']['Authorization'], 'Bearer ' + TOKEN)
        self.assertEqual(calls[0][0], 'ssh')
        self.assertEqual(calls[0][3], 'me@macbook')
        self.assertEqual(calls[1][0], 'scp')
        self.assertEqual(calls[2][0], 'ffmpeg')
        self.assertEqual(calls[3][0], 'ssh')
        self.assertIn('rm -rf /tmp/scripter-youtube-ingest-', calls[3][-1])

    def test_error_body_and_ids_do_not_echo_the_token(self):
        def transport(method, url, data=None, headers=None, timeout=60):
            return 401, {}, json.dumps({'detail': 'rejected ' + TOKEN}).encode()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.m4a'
            path.write_bytes(b'abc')
            with self.assertRaises(IngestError) as caught:
                post_audio(path, TOKEN, transport=transport)
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertIn('***', str(caught.exception))
        self.assertIn('401', str(caught.exception))

        def ok_but_secret(method, url, data=None, headers=None, timeout=60):
            return 200, {}, json.dumps({'project_id': TOKEN, 'conversation_id': 'c'}).encode()

        def run(args, **kwargs):
            template = args[args.index('-o') + 1]
            path = Path(template.replace('%(title)s', 't').replace('%(ext)s', 'm4a'))
            path.write_bytes(b'abc')
            return completed(args, stdout=str(path) + '\n')

        with tempfile.TemporaryDirectory() as folder:
            _path, lines = ingest_url(
                URL, output_dir=folder, token=TOKEN, transport=ok_but_secret, run=run, which=tools)
        self.assertNotIn(TOKEN, '\n'.join(lines))
        self.assertIn('project_id: ***', lines)

    def test_missing_ids_include_a_redacted_body(self):
        def transport(method, url, data=None, headers=None, timeout=60):
            return 200, {}, json.dumps({'ok': True, 'note': TOKEN}).encode()

        def run(args, **kwargs):
            template = args[args.index('-o') + 1]
            path = Path(template.replace('%(title)s', 't').replace('%(ext)s', 'm4a'))
            path.write_bytes(b'abc')
            return completed(args, stdout=str(path) + '\n')

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(IngestError) as caught:
                ingest_url(URL, output_dir=folder, token=TOKEN, transport=transport, run=run, which=tools)
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertIn('***', str(caught.exception))

    def test_real_http_stack_posts_multipart_to_a_local_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), _IngestHandler)
        server.seen = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / '강의, 1부.m4a'
                path.write_bytes(b'0123456789')
                payload = post_audio(
                    path, TOKEN,
                    base_url=f'http://127.0.0.1:{server.server_address[1]}',
                    transport=request,
                )
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertEqual(payload['project_id'], 'proj-1')
        self.assertEqual(payload['conversation_id'], 'conv-1')
        seen = server.seen
        self.assertEqual(seen['path'], '/api/ingest/audio')
        self.assertEqual(seen['authorization'], 'Bearer ' + TOKEN)
        self.assertNotIn(TOKEN.encode(), seen['body'])
        parts = multipart_parts(seen['body'], seen['content_type'])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][1], b'false')
        self.assertEqual(parts[1][1], b'0123456789')
        self.assertEqual(filename_as_server_sees_it(parts[1][0]), '강의, 1부.m4a')

    def test_non_m4a_is_left_alone(self):
        def run(args, **kwargs):
            template = args[args.index('-o') + 1]
            path = Path(template.replace('%(title)s', 'clip').replace('%(ext)s', 'webm'))
            path.write_bytes(b'webm')
            return completed(args, stdout=str(path) + '\n')

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(IngestError) as caught:
                download_local(URL, folder, run=run, which=tools)
        self.assertIn('다시 인코딩하지 않습니다', str(caught.exception))
        self.assertNotIn('ffmpeg', str(caught.exception))

    def test_bot_check_on_this_machine_names_the_ssh_option(self):
        def run(args, **kwargs):
            return completed(args, code=1, stderr='ERROR: [youtube] Sign in to confirm you’re not a bot')

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(IngestError) as caught:
                download_local(URL, folder, run=run, which=tools)
        self.assertIn('--via-ssh', str(caught.exception))

    def test_run_streaming_captures_stdout(self):
        completed_process = run_streaming([
            sys.executable, '-c',
            'import sys; sys.stderr.write("err-line\\n"); print("out-line")',
        ])
        self.assertEqual(completed_process.returncode, 0)
        self.assertIn('out-line', completed_process.stdout)
        self.assertIn('err-line', completed_process.stderr)

    def test_main_prints_ids(self):
        with patch('youtube_ingest.ingest_url', return_value=(Path('files/a.m4a'), [
            'project_id: p1', 'conversation_id: c1',
        ])):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main([URL])
        text = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn('project_id: p1', text)
        self.assertIn('conversation_id: c1', text)
        self.assertNotIn(TOKEN, text)

    def test_main_exits_with_the_error_and_not_a_traceback_secret(self):
        with patch('youtube_ingest.ingest_url', side_effect=IngestError('봇 확인')):
            with self.assertRaises(SystemExit) as caught:
                main([URL])
        self.assertIn('봇 확인', str(caught.exception))
        self.assertNotIn(TOKEN, str(caught.exception))


class ParseTests(unittest.TestCase):
    def test_via_ssh_flag(self):
        args = parse_args([URL, '--via-ssh', 'me@macbook', '--output-dir', 'files'])
        self.assertEqual(args.via_ssh, 'me@macbook')
        self.assertEqual(args.output_dir, Path('files'))
        self.assertIsNone(args.token_file)
        self.assertTrue(args.base_url.startswith('https://markvector.fly.dev'))


if __name__ == '__main__':
    unittest.main()
