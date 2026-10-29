"""Download a YouTube m4a and POST it to mark-vector.

The native audio stream is uploaded as-is. Nothing is re-encoded, and the
audio is not summarized. enqueue=false, so this does not enter the worker queue.

Contract:
  Authorization: Bearer <token from ~/.config/markvector/ingest_token>
  POST {base}/api/ingest/audio
    multipart field `file` (the m4a, filename is the video title)
    form field enqueue=false

The token is read at runtime and is never printed, logged, or written back.
A title with a comma stays inside the quoted filename parameter. curl -F
treats that comma as a separator; this body does not.

YouTube's bot check sometimes blocks the machine running this script. --via-ssh
downloads with yt-dlp on another Mac (Homebrew yt-dlp, ffmpeg not required
there), copies the file back, and remuxes it here with `ffmpeg -c copy`.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import urllib.parse
import uuid
from pathlib import Path

from net import error_text, request


DEFAULT_BASE = 'https://markvector.fly.dev'
INGEST_PATH = '/api/ingest/audio'
DEFAULT_OUTPUT_DIR = Path('files')
# One audio-only m4a stream. No bestaudio fallback: that can be webm/opus,
# and this script does not re-encode.
AUDIO_FORMAT = 'bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]'
ENQUEUE = 'false'
CONNECT_TIMEOUT = '20'
_REMOTE_DIR = re.compile(r'^/tmp/scripter-youtube-ingest-[0-9a-f]{32}$')
_SSH_HOST = re.compile(r'^(?:[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?@)?[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$')
_EXT = re.compile(r'[A-Za-z0-9]{1,8}')
_TOKEN_NAME = 'ingest_token'


class IngestError(RuntimeError):
    pass


def default_token_path():
    return Path.home() / '.config' / 'markvector' / _TOKEN_NAME


def redact(text, secret):
    if not secret or text is None:
        return '' if text is None else str(text)
    return str(text).replace(str(secret), '***')


def audio_filename(title, ext='m4a'):
    """A single path component. Commas and quotes stay; slashes do not."""
    suffix = str(ext or 'm4a').lstrip('.').lower()
    if not re.fullmatch(r'[a-z0-9]{1,8}', suffix):
        suffix = 'm4a'
    parts = []
    for part in re.split(r'[/\\]+', str(title or '')):
        part = re.sub(r'[\x00-\x1f\x7f]', ' ', part)
        part = re.sub(r'\s+', ' ', part).strip().strip('.')
        if part in ('', '.', '..'):
            continue
        parts.append(part)
    text = ' '.join(parts).strip()
    if not text:
        text = 'audio'
    if len(text) > 180:
        text = text[:180].rstrip() or 'audio'
    return f'{text}.{suffix}'


def unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f'{path.stem} {number}{path.suffix}')
        if not candidate.exists():
            return candidate
    raise IngestError(f'같은 이름의 파일이 너무 많습니다: {path.name}')


def validate_youtube_url(url):
    text = str(url or '').strip()
    if not text or any(ch in text for ch in '\r\n\x00 \t'):
        raise IngestError('유튜브 주소가 필요합니다.')
    parts = urllib.parse.urlsplit(text)
    host = (parts.hostname or '').lower().rstrip('.')
    if host.startswith('www.'):
        host = host[4:]
    youtube = (
        host in ('youtu.be', 'youtube.com', 'youtube-nocookie.com')
        or host.endswith('.youtube.com')
        or host.endswith('.youtube-nocookie.com')
    )
    if parts.scheme not in ('http', 'https') or not youtube:
        raise IngestError('http(s) 유튜브 주소만 받을 수 있습니다.')
    return text


def validate_ssh_host(host):
    text = str(host or '').strip()
    if not _SSH_HOST.fullmatch(text):
        raise IngestError(
            'SSH 호스트는 user@host 또는 host 형식이어야 합니다. '
            '포트가 필요하면 ~/.ssh/config에 적으세요.')
    return text


def normalize_base(url):
    text = (url or DEFAULT_BASE).strip().rstrip('/')
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise IngestError('mark-vector 주소는 http(s)여야 합니다.')
    return text


def ytdlp_args(url, output_template, prints):
    """yt-dlp argv. No -x and no re-encode; the container stays m4a."""
    args = [
        'yt-dlp',
        '--no-playlist',
        '--no-progress',
        '--no-mtime',
        # --print would otherwise skip the download when WHEN is early.
        '--no-simulate',
        '-f', AUDIO_FORMAT,
        '-o', output_template,
    ]
    for item in prints:
        args.extend(['--print', item])
    args.append(url)
    return args


def remote_shell(url, remote_dir):
    """A POSIX command for a Mac that has brew yt-dlp and may lack ffmpeg.

    PATH is extended because a non-interactive ssh does not load Homebrew.
    The file is `audio.<ext>` so the path copied back has no spaces.
    """
    if not _REMOTE_DIR.fullmatch(remote_dir):
        raise IngestError('원격 임시 경로가 올바르지 않습니다.')
    template = remote_dir + '/audio.%(ext)s'
    download = shlex.join(ytdlp_args(url, template, ['after_move:title', 'after_move:filepath']))
    return ' && '.join([
        'export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"',
        f'mkdir -p {shlex.quote(remote_dir)}',
        download,
    ])


def ssh_argv(host, remote_command):
    return ['ssh', '-o', 'ConnectTimeout=' + CONNECT_TIMEOUT, host, remote_command]


def scp_argv(host, remote_path, local_path):
    return ['scp', '-o', 'ConnectTimeout=' + CONNECT_TIMEOUT, f'{host}:{remote_path}', str(local_path)]


def ffmpeg_remux_args(source, destination):
    """Remux only. -c copy does not re-encode."""
    return [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(source), '-c', 'copy', str(destination),
    ]


def require_tool(name, which=shutil.which):
    if which(name) is None:
        brew = 'yt-dlp' if name == 'yt-dlp' else name
        raise IngestError(f'필요한 프로그램이 없습니다: {name}. brew install {brew}')
    return name


def run_streaming(args, stdout=None, stderr=None, text=False, **kwargs):
    """Run a command, stream stderr to the terminal, and keep both streams.

    ssh writes its password prompt to stderr. yt-dlp progress and errors do too.
    stdout is captured because it is the machine-readable --print output.
    """
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    err_lines = []

    def pump():
        for line in process.stderr:
            err_lines.append(line)
            print(line, end='', file=sys.stderr, flush=True)

    thread = threading.Thread(target=pump)
    thread.start()
    try:
        out = process.stdout.read()
        returncode = process.wait()
    finally:
        thread.join()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(args, returncode, out, ''.join(err_lines))


def run_command(run, args, check=True, via_ssh=False):
    try:
        completed = run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as error:
        raise IngestError(f'{args[0]} 실행 실패: {error}') from error
    returncode = getattr(completed, 'returncode', 0)
    stdout = getattr(completed, 'stdout', '') or ''
    stderr = getattr(completed, 'stderr', '') or ''
    if check and returncode != 0:
        raise command_failed(args[0], returncode, stderr or stdout, via_ssh=via_ssh)
    return completed


def command_failed(tool, returncode, detail, via_ssh=False):
    text = ' '.join(str(detail or '').split())
    lowered = text.lower().replace('’', "'").replace('‘', "'")
    if 'sign in to confirm' in lowered and 'bot' in lowered:
        if via_ssh:
            message = '원격 맥에서도 YouTube 봇 확인에 막혔습니다.'
        else:
            message = 'YouTube 봇 확인에 막혔습니다. 다른 맥에서 받으려면 --via-ssh <host>를 붙이세요.'
        if text:
            message = f'{message} {text[:400]}'
        return IngestError(message)
    if 'command not found' in lowered and 'yt-dlp' in lowered:
        where = '원격 맥' if tool == 'ssh' or via_ssh else '이 컴퓨터'
        return IngestError(f'{where}에 yt-dlp가 없습니다. brew install yt-dlp')
    if 'command not found' in lowered and 'ffmpeg' in lowered:
        return IngestError('ffmpeg가 없습니다. brew install ffmpeg')
    if len(text) > 800:
        text = text[:800] + '…'
    return IngestError(f'{tool} 실패 (종료 {returncode}): {text or "출력 없음"}')


def parse_local_path(stdout):
    lines = [line.strip() for line in str(stdout or '').splitlines() if line.strip()]
    if not lines:
        raise IngestError('yt-dlp가 파일 경로를 출력하지 않았습니다.')
    return Path(lines[-1])


def parse_remote_output(stdout, remote_dir):
    lines = [line.strip('\r') for line in str(stdout or '').splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 2:
        raise IngestError('원격 yt-dlp 출력에서 제목과 경로를 찾지 못했습니다.')
    remote_path, ext = remote_audio_path(remote_dir, lines[-1])
    title = '\n'.join(line.strip() for line in lines[:-1]).strip()
    if not title:
        raise IngestError('원격 yt-dlp가 제목을 출력하지 않았습니다.')
    return title, remote_path, ext


def remote_audio_path(remote_dir, printed):
    printed = str(printed or '').strip()
    prefix = remote_dir + '/audio.'
    if not printed.startswith(prefix):
        raise IngestError('원격 yt-dlp가 예상한 경로를 출력하지 않았습니다.')
    ext = printed[len(prefix):]
    if not _EXT.fullmatch(ext):
        raise IngestError(f'원격 파일 확장자가 올바르지 않습니다: {ext!r}')
    return printed, ext


def contained_file(path, root):
    path = Path(path)
    root = Path(root).resolve()
    if path.is_symlink():
        raise IngestError(f'심볼릭 링크는 올리지 않습니다: {path}')
    if not path.is_file():
        raise IngestError(f'받은 파일이 없습니다: {path}')
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise IngestError(f'받은 파일이 출력 폴더 밖입니다: {path}') from error
    if resolved.stat().st_size <= 0:
        raise IngestError(f'받은 파일이 비어 있습니다: {resolved.name}')
    return resolved


def ensure_output_dir(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise IngestError(f'출력 폴더를 만들지 못했습니다: {path} ({error.strerror or error})') from error
    if not path.is_dir():
        raise IngestError(f'출력 경로가 폴더가 아닙니다: {path}')
    return path


def download_local(url, output_dir, run=run_streaming, which=shutil.which):
    require_tool('yt-dlp', which)
    output_dir = ensure_output_dir(output_dir)
    template = str(output_dir / '%(title)s.%(ext)s')
    print(f'받는 중: {url}', flush=True)
    completed = run_command(run, ytdlp_args(url, template, ['after_move:filepath']))
    path = contained_file(parse_local_path(completed.stdout), output_dir)
    if path.suffix.lower() != '.m4a':
        raise IngestError(f'm4a가 아닙니다: {path.name}. 다시 인코딩하지 않습니다.')
    print(f'받음: {path}', flush=True)
    return path


def download_via_ssh(url, host, output_dir, run=run_streaming, which=shutil.which):
    """Download on `host` (no ffmpeg there), then remux here with -c copy."""
    host = validate_ssh_host(host)
    require_tool('ffmpeg', which)
    output_dir = ensure_output_dir(output_dir)
    remote_dir = '/tmp/scripter-youtube-ingest-' + uuid.uuid4().hex
    print(f'SSH로 받는 중: {host}', flush=True)
    local_temp = None
    try:
        completed = run_command(
            run, ssh_argv(host, remote_shell(url, remote_dir)), via_ssh=True)
        title, remote_path, ext = parse_remote_output(completed.stdout, remote_dir)
        local_temp = output_dir / f'.download-{uuid.uuid4().hex}.{ext.lower()}'
        run_command(run, scp_argv(host, remote_path, local_temp), via_ssh=True)
        downloaded = contained_file(local_temp, output_dir)
        named = unique_path(output_dir / audio_filename(title, ext))
        downloaded.replace(named)
        local_temp = None
        print(f'리먹스: {named.name}', flush=True)
        return remux_copy(named, run=run, which=which)
    finally:
        if local_temp is not None:
            Path(local_temp).unlink(missing_ok=True)
        try:
            run_command(run, ssh_argv(host, f'rm -rf {shlex.quote(remote_dir)}'), check=False)
        except IngestError as error:
            print(f'원격 임시 폴더를 지우지 못했습니다: {error}', flush=True)


def remux_copy(source, run=run_streaming, which=shutil.which):
    require_tool('ffmpeg', which)
    source = Path(source)
    temporary = source.with_name(f'.remux-{uuid.uuid4().hex}.m4a')
    final = source.with_suffix('.m4a')
    try:
        run_command(run, ffmpeg_remux_args(source, temporary))
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise IngestError(f'리먹스 결과가 비어 있습니다: {source.name}')
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if source.resolve() != final.resolve():
        source.unlink(missing_ok=True)
    final = contained_file(final, source.parent)
    if final.suffix.lower() != '.m4a':
        raise IngestError(f'리먹스 결과가 m4a가 아닙니다: {final.name}')
    return final


def read_token(path=None):
    """Read the bearer token. The value is not printed."""
    path = Path(path) if path else default_token_path()
    try:
        info = path.lstat()
    except OSError as error:
        raise IngestError(
            f'ingest 토큰을 읽을 수 없습니다: {path} ({error.strerror or error}). '
            '한 줄짜리 토큰 파일을 두고 chmod 600 하세요.') from error
    if stat.S_ISLNK(info.st_mode):
        raise IngestError(f'ingest 토큰 파일이 심볼릭 링크입니다: {path}')
    if not stat.S_ISREG(info.st_mode):
        raise IngestError(f'ingest 토큰 파일이 아닙니다: {path}')
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        print(f'경고: 토큰 파일이 다른 사용자에게 열려 있습니다 ({oct(mode)}). chmod 600 {path}', flush=True)
    try:
        raw = path.read_text(encoding='utf-8')
    except OSError as error:
        raise IngestError(f'ingest 토큰을 읽을 수 없습니다: {path} ({error.strerror or error})') from error
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise IngestError(f'ingest 토큰 파일은 비어 있지 않은 한 줄이어야 합니다: {path}')
    token = lines[0]
    if len(token) > 4096:
        raise IngestError('ingest 토큰이 너무 깁니다.')
    return token


def quote_filename(filename):
    """RFC 7578 quoted filename. Commas and semicolons stay inside the quotes.

    python-multipart ignores filename* (RFC 7578 forbids it in multipart) and
    decodes the header as latin-1, then the app decodes those bytes as UTF-8.
    The bytes of this quoted string are therefore the UTF-8 filename.
    """
    text = str(filename).replace('\r', ' ').replace('\n', ' ')
    escaped = text.replace('\\', '\\\\').replace('"', '\\"')
    return '"' + escaped + '"'


def encode_multipart(filename, content, fields, boundary_factory=None):
    content = bytes(content)
    if boundary_factory is None:
        def boundary_factory():
            return '----scripter' + uuid.uuid4().hex
    for key, value in fields:
        if any(ch in f'{key}{value}' for ch in '\r\n"'):
            raise IngestError('폼 필드에 줄바꿈이나 따옴표가 있습니다.')
    filename_bytes = str(filename).encode('utf-8')
    boundary = None
    for _ in range(8):
        candidate = boundary_factory()
        if not re.fullmatch(r'[A-Za-z0-9-]{1,70}', candidate):
            raise IngestError('multipart 경계가 올바르지 않습니다.')
        raw = candidate.encode('ascii')
        if raw not in content and raw not in filename_bytes:
            boundary = candidate
            break
    if boundary is None:
        raise IngestError('multipart 경계를 만들지 못했습니다.')
    chunks = []
    for key, value in fields:
        header = f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        chunks.append(f'--{boundary}\r\n'.encode('ascii') + header.encode('utf-8'))
    disposition = f'Content-Disposition: form-data; name="file"; filename={quote_filename(filename)}\r\n'
    file_header = (
        f'--{boundary}\r\n'.encode('ascii')
        + disposition.encode('utf-8')
        + b'Content-Type: audio/mp4\r\n\r\n'
    )
    closing = f'\r\n--{boundary}--\r\n'.encode('ascii')
    return b''.join(chunks) + file_header + content + closing, f'multipart/form-data; boundary={boundary}'


def build_multipart(path, enqueue=ENQUEUE):
    path = Path(path)
    return encode_multipart(path.name, path.read_bytes(), (('enqueue', enqueue),))


def post_audio(path, token, base_url=DEFAULT_BASE, transport=request, timeout=1800):
    token = str(token or '')
    if not token.strip() or '\n' in token or '\r' in token:
        raise IngestError('ingest 토큰이 비어 있거나 줄바꿈이 있습니다.')
    path = Path(path)
    if path.suffix.lower() != '.m4a' or not path.is_file() or path.stat().st_size <= 0:
        raise IngestError(f'올릴 m4a가 없습니다: {path}')
    body, content_type = build_multipart(path)
    url = normalize_base(base_url) + INGEST_PATH
    headers = {
        'Authorization': 'Bearer ' + token,
        'Accept': 'application/json',
        'Content-Type': content_type,
        'Content-Length': str(len(body)),
    }
    print(f'업로드 중: {path.name}', flush=True)
    try:
        try:
            status, _headers, raw = transport('POST', url, data=body, headers=headers, timeout=timeout)
        except TypeError:
            status, _headers, raw = transport('POST', url, body, headers, timeout)
    except IngestError:
        raise
    except Exception as error:
        message = redact(str(error), token)
        if token in str(error):
            raise IngestError(f'인제스트 요청 실패: {message}') from None
        if isinstance(error, (TimeoutError, OSError)):
            raise IngestError(f'인제스트 요청 실패: {error}') from error
        raise
    if isinstance(raw, bytes):
        text = raw.decode('utf-8', 'replace')
    else:
        text = '' if raw is None else str(raw)
    safe = redact(text, token)
    if status not in (200, 201, 202):
        raise IngestError(f'오디오 인제스트 실패 (HTTP {status}): {error_text(safe)}')
    try:
        payload = json.loads(text) if text.strip() else {}
    except ValueError as error:
        raise IngestError('인제스트 응답이 JSON이 아닙니다: ' + safe[:500]) from error
    if not isinstance(payload, dict):
        raise IngestError('인제스트 응답이 객체가 아닙니다: ' + safe[:500])
    return payload


def _add_id(bucket, value):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, list):
        for item in value:
            _add_id(bucket, item)
        return
    if isinstance(value, dict):
        if 'id' in value:
            _add_id(bucket, value.get('id'))
        return
    text = str(value).strip()
    if text and text not in bucket:
        bucket.append(text)


def extract_ids(payload):
    """project_id and conversation_id, including nested project/conversation objects."""
    projects = []
    conversations = []
    if not isinstance(payload, dict):
        return {'project_ids': projects, 'conversation_ids': conversations}

    def walk(node, depth=0):
        if depth > 8 or node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth + 1)
            return
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in ('project_id', 'projectId', 'project'):
                _add_id(projects, value)
            elif key in (
                'conversation_id', 'conversationId', 'conversation_ids', 'conversationIds',
                'conversation', 'conversations',
            ):
                _add_id(conversations, value)
            walk(value, depth + 1)

    walk(payload)
    if not conversations:
        candidate = payload.get('id')
        if not isinstance(candidate, (dict, list, bool)) and candidate is not None:
            text = str(candidate).strip()
            if text and text not in projects:
                conversations.append(text)
    return {'project_ids': projects, 'conversation_ids': conversations}


def format_id_lines(payload):
    found = extract_ids(payload)
    lines = [f'project_id: {item}' for item in found['project_ids']]
    lines.extend(f'conversation_id: {item}' for item in found['conversation_ids'])
    return lines


def ingest_url(url, via_ssh=None, output_dir=DEFAULT_OUTPUT_DIR, token=None, token_path=None,
               base_url=DEFAULT_BASE, transport=request, run=run_streaming, which=shutil.which):
    url = validate_youtube_url(url)
    if via_ssh:
        path = download_via_ssh(url, via_ssh, output_dir, run=run, which=which)
    else:
        path = download_local(url, output_dir, run=run, which=which)
    secret = token if token is not None else read_token(token_path)
    payload = post_audio(path, secret, base_url=base_url, transport=transport)
    lines = [redact(line, secret) for line in format_id_lines(payload)]
    if not lines:
        body = redact(json.dumps(payload, ensure_ascii=False), secret)
        if len(body) > 2000:
            body = body[:2000] + '…'
        raise IngestError('응답에서 project id와 conversation id를 찾지 못했습니다: ' + body)
    return path, lines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='youtube_ingest.py',
        description='유튜브 주소의 m4a 음성을 받아 mark-vector에 넣습니다. 요약하거나 다시 인코딩하지 않습니다.',
    )
    parser.add_argument('url', help='유튜브 주소')
    parser.add_argument('--via-ssh', metavar='HOST', default=None,
                        help='이 컴퓨터가 봇 확인에 막히면 HOST에서 yt-dlp로 받은 뒤, 여기서 ffmpeg -c copy로 리먹스')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='받은 m4a를 둘 폴더 (기본 files)')
    parser.add_argument('--base-url', default=DEFAULT_BASE,
                        help=f'mark-vector 주소 (기본 {DEFAULT_BASE})')
    parser.add_argument('--token-file', type=Path, default=None,
                        help='Bearer 토큰 파일 (기본 ~/.config/markvector/ingest_token). 출력하지 않습니다.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        path, lines = ingest_url(
            args.url,
            via_ssh=args.via_ssh,
            output_dir=args.output_dir,
            token_path=args.token_file,
            base_url=args.base_url,
        )
    except IngestError as error:
        raise SystemExit(str(error)) from None
    print(f'오디오: {path}', flush=True)
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == '__main__':
    main()
