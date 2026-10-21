"""YouTube Data API v3: desktop OAuth and a resumable videos.insert upload.

Default privacy is private. An unverified API project can only upload private
videos, and the channel owner publishes from YouTube Studio. A thumbnail
rejection (the channel is often unverified) does not undo the video upload.
"""
import base64
import hashlib
import json
import os
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from net import atomic_write, error_text, header, request


SCOPES = (
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.force-ssl',
)
UPLOAD_URL = 'https://www.googleapis.com/upload/youtube/v3/videos'
THUMBNAIL_URL = 'https://www.googleapis.com/upload/youtube/v3/thumbnails/set'
AUTH_TIMEOUT = 180
CHUNK = 8 * 1024 * 1024
ROOT = Path(__file__).resolve().parent


def category_id(value=None):
    """YouTube video category. 27 is Education."""
    chosen = value or os.environ.get('YOUTUBE_CATEGORY_ID', '').strip() or '27'
    if not str(chosen).isdigit():
        raise RuntimeError('YOUTUBE_CATEGORY_ID는 숫자여야 합니다.')
    return str(chosen)


def resolve_client_secret(path=None):
    if path:
        return Path(path)
    env = os.environ.get('YOUTUBE_CLIENT_SECRET', '').strip()
    if env:
        return Path(env)
    direct = ROOT / 'client_secret.json'
    if direct.is_file():
        return direct
    found = sorted(ROOT.glob('client_secret*.json'))
    if len(found) == 1:
        return found[0]
    if found:
        raise RuntimeError('client_secret JSON이 여러 개입니다. --client-secret으로 고르세요.')
    raise RuntimeError(
        'OAuth 클라이언트 JSON이 없습니다. Google Cloud에서 데스크톱 앱 클라이언트를 만들어 '
        'client_secret.json으로 저장하세요.')


def resolve_token_file(path=None):
    if path:
        return Path(path)
    env = os.environ.get('YOUTUBE_TOKEN_FILE', '').strip()
    if env:
        return Path(env)
    return ROOT / '.youtube-token.json'


def load_client(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f'OAuth 클라이언트 JSON을 읽을 수 없습니다: {path} ({error})') from error
    config = data.get('installed') or data.get('web') if isinstance(data, dict) else None
    if not isinstance(config, dict):
        raise RuntimeError('클라이언트 JSON에 installed(데스크톱 앱) 항목이 없습니다.')
    for key in ('client_id', 'client_secret', 'auth_uri', 'token_uri'):
        if not config.get(key):
            raise RuntimeError(f'클라이언트 JSON에 {key}가 없습니다.')
    return config


def pkce_pair():
    verifier = base64.urlsafe_b64encode(os.urandom(32)).decode('ascii').rstrip('=')
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    challenge = base64.urlsafe_b64encode(digest).decode('ascii').rstrip('=')
    return verifier, challenge


def authorization_url(config, redirect_uri, state, challenge):
    query = urllib.parse.urlencode({
        'client_id': config['client_id'],
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
        'include_granted_scopes': 'true',
        'state': state,
        'code_challenge': challenge,
        'code_challenge_method': 'S256',
    })
    return config['auth_uri'] + ('&' if '?' in config['auth_uri'] else '?') + query


def parse_callback(query, expected_state):
    if query.get('state', [None])[0] != expected_state:
        raise RuntimeError('OAuth state가 일치하지 않습니다. 인증을 다시 시작하세요.')
    if query.get('error'):
        description = (query.get('error_description') or query.get('error'))[0]
        raise RuntimeError(f'유튜브 인증이 거절되었습니다: {description}')
    code = (query.get('code') or [None])[0]
    if not code:
        raise RuntimeError('인증 코드가 없습니다.')
    return code


class _Callback(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ('/', '/callback') and (query.get('code') or query.get('error')):
            self.server.result = query
        body = '인증이 끝났습니다. 이 창을 닫고 터미널로 돌아가세요.\n'.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def wait_for_callback(server, timeout):
    server.timeout = 1
    deadline = time.time() + timeout
    while server.result is None and time.time() < deadline:
        server.handle_request()
    if server.result is None:
        raise RuntimeError('제한 시간 안에 브라우저 인증이 끝나지 않았습니다.')
    return server.result


def description_with_chapters(metadata, limit=5000):
    """Description plus chapter lines, the same shape as .youtube.txt.

    Chapters are dropped from the end if they would pass YouTube's 5000
    characters. The description itself is cut only when it alone is longer.
    """
    description = str(metadata.get('description') or '').strip()
    lines = []
    for chapter in metadata.get('chapters') or []:
        if not isinstance(chapter, dict):
            continue
        start = str(chapter.get('start') or '').strip()
        title = ' '.join(str(chapter.get('title') or '').split())
        if start and title:
            lines.append(f'{start} {title}')
    while lines:
        extra = '\n'.join(lines)
        combined = f'{description}\n\n{extra}' if description else extra
        if len(combined) <= limit:
            return combined
        lines.pop()
    return description[:limit]


def video_body(metadata, privacy='private', category='27'):
    """videos.insert snippet and status. Language is Korean."""
    if privacy not in ('private', 'unlisted', 'public'):
        raise RuntimeError('privacyStatus는 private, unlisted, public 중 하나여야 합니다.')
    if not isinstance(metadata, dict):
        raise RuntimeError('유튜브 자료(.youtube.json)가 잘못되었습니다.')
    title = ' '.join(str(metadata.get('title') or '').split())
    if not title:
        raise RuntimeError('유튜브 제목이 비어 있습니다.')
    if len(title) > 100:
        title = title[:100].rstrip()
    return {
        'snippet': {
            'title': title,
            'description': description_with_chapters(metadata),
            'categoryId': category_id(category),
            'defaultLanguage': 'ko',
            'defaultAudioLanguage': 'ko',
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': False,
        },
    }


def marker_path(video):
    return Path(video).with_suffix('.youtube-upload.json')


def _load_marker(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_marker(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


class Credentials:
    """Refresh token on disk (mode 600). Access tokens are cached until expiry."""

    def __init__(self, secret_path, token_path, transport=request, sleep=time.sleep):
        self.secret_path = Path(secret_path)
        self.token_path = Path(token_path)
        self.transport = transport
        self.sleep = sleep
        self.config = None

    def _config(self):
        if self.config is None:
            self.config = load_client(self.secret_path)
        return self.config

    def load(self):
        path = self.token_path
        if path.is_symlink():
            raise RuntimeError(f'토큰 파일이 심볼릭 링크입니다: {path}')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError(
                '유튜브 갱신 토큰이 없습니다. 터미널에서 uv run scripter.py --youtube-auth 를 실행하세요. '
                f'({error})') from error
        if not isinstance(data, dict) or not data.get('refresh_token'):
            raise RuntimeError('유튜브 갱신 토큰이 없습니다. uv run scripter.py --youtube-auth 를 실행하세요.')
        return data

    def ready(self):
        try:
            data = self.load()
        except RuntimeError:
            return False
        return bool(data.get('client_id'))

    def _store(self, payload, previous=None):
        config = self._config()
        refresh = payload.get('refresh_token') or (previous or {}).get('refresh_token')
        if not refresh:
            raise RuntimeError('갱신 토큰을 받지 못했습니다. uv run scripter.py --youtube-auth 로 다시 동의하세요.')
        if not payload.get('access_token'):
            raise RuntimeError('액세스 토큰을 받지 못했습니다.')
        expires = time.time() + float(payload.get('expires_in') or 3600) - 60
        data = {
            'refresh_token': refresh,
            'access_token': payload['access_token'],
            'expires_at': expires,
            'token_type': payload.get('token_type') or 'Bearer',
            'scope': payload.get('scope') or ' '.join(SCOPES),
            'client_id': config['client_id'],
        }
        atomic_write(self.token_path, json.dumps(data, indent=2) + '\n')
        return data

    def post_form(self, form):
        config = self._config()
        body = urllib.parse.urlencode(form).encode('utf-8')
        status, _, raw = self.transport(
            'POST', config['token_uri'], data=body,
            headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
            timeout=60)
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except (UnicodeError, ValueError) as error:
            raise RuntimeError(f'토큰 응답을 읽을 수 없습니다 (HTTP {status}).') from error
        if status != 200 or not isinstance(payload, dict) or payload.get('error'):
            message = payload.get('error_description') or payload.get('error') if isinstance(payload, dict) else error_text(raw)
            raise RuntimeError(f'유튜브 토큰 요청 실패 (HTTP {status}): {message}')
        return payload

    def exchange(self, code, redirect_uri, verifier):
        config = self._config()
        payload = self.post_form({
            'code': code,
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code',
            'code_verifier': verifier,
        })
        return self._store(payload)

    def refresh(self, previous=None):
        previous = previous or self.load()
        config = self._config()
        if previous.get('client_id') and previous['client_id'] != config['client_id']:
            raise RuntimeError('토큰이 다른 OAuth 클라이언트의 것입니다. --youtube-auth로 다시 동의하세요.')
        payload = self.post_form({
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'refresh_token': previous['refresh_token'],
            'grant_type': 'refresh_token',
        })
        return self._store(payload, previous)

    def authorization(self):
        data = self.load()
        if data.get('access_token') and float(data.get('expires_at') or 0) > time.time():
            return 'Bearer ' + data['access_token']
        refreshed = self.refresh(data)
        return 'Bearer ' + refreshed['access_token']

    def force_refresh(self):
        refreshed = self.refresh(self.load())
        return 'Bearer ' + refreshed['access_token']


def authorize_interactive(client_secret=None, token_file=None, open_browser=None, timeout=AUTH_TIMEOUT, transport=request):
    """Open a browser, receive the loopback redirect, and store a refresh token."""
    secret = resolve_client_secret(client_secret)
    token = resolve_token_file(token_file)
    config = load_client(secret)
    credentials = Credentials(secret, token, transport=transport)
    credentials.config = config
    verifier, challenge = pkce_pair()
    state = base64.urlsafe_b64encode(os.urandom(16)).decode('ascii').rstrip('=')
    server = HTTPServer(('127.0.0.1', 0), _Callback)
    server.result = None
    port = server.server_address[1]
    redirect = f'http://127.0.0.1:{port}/'
    url = authorization_url(config, redirect, state, challenge)
    print('브라우저에서 유튜브 업로드를 허용해 주세요.', flush=True)
    print(url, flush=True)
    opener = webbrowser.open if open_browser is None else open_browser
    try:
        opened = bool(opener(url))
    except Exception as error:
        opened = False
        print(f'브라우저를 열지 못했습니다 ({error}). 위 주소를 직접 여세요.', flush=True)
    if not opened:
        print('위 주소가 열리지 않으면 브라우저에 붙여 넣으세요.', flush=True)
    try:
        query = wait_for_callback(server, timeout)
    finally:
        server.server_close()
    code = parse_callback(query, state)
    saved = credentials.exchange(code, redirect, verifier)
    print(f'갱신 토큰 저장: {token.resolve()}', flush=True)
    return saved


def _range_end(headers):
    value = header(headers, 'Range') or ''
    matched = None
    for part in str(value).split(','):
        if '-' not in part:
            continue
        try:
            matched = int(part.split('-', 1)[1])
        except ValueError:
            continue
    if matched is None:
        return 0
    return matched + 1


class UploadFinished(Exception):
    def __init__(self, video):
        self.video = video


class YouTubeClient:
    def __init__(self, credentials, transport=request, chunk_size=CHUNK, sleep=time.sleep):
        self.credentials = credentials
        self.transport = transport
        self.chunk_size = chunk_size
        self.sleep = sleep

    def _call(self, method, url, data, headers, timeout):
        try:
            return self.transport(method, url, data=data, headers=headers, timeout=timeout)
        except TypeError:
            return self.transport(method, url, data, headers, timeout)

    def upload(self, video_path, metadata, privacy='private', category=None, on_progress=None):
        """Resumable videos.insert, then thumbnails.set. Thumbnail errors are returned, not raised."""
        video_path = Path(video_path)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f'업로드할 영상이 없습니다: {video_path}')
        body = video_body(metadata, privacy=privacy, category=category)
        size = video_path.stat().st_size
        marker = marker_path(video_path)
        saved = _load_marker(marker)
        video = None
        if saved.get('youtube_video_id'):
            video = {'id': saved['youtube_video_id']}
        else:
            session = saved.get('upload_url') if saved.get('bytes') == size else None
            video, session = self._send_video(video_path, body, size, session, on_progress)
            saved = {
                'bytes': size,
                'upload_url': session,
                'youtube_video_id': video['id'],
                'youtube_url': watch_url(video['id']),
                'privacy': privacy,
                'thumbnail': False,
            }
            _save_marker(marker, saved)
        video_id = video['id']
        url = watch_url(video_id)
        thumbnail = saved.get('thumbnail') is True
        image = video_path.with_suffix('.thumbnail.png')
        if not image.is_file():
            image = video_path.parent / (video_path.stem + '.thumbnail.png')
        if image.is_file() and not thumbnail:
            thumbnail = self._thumbnail(video_id, image)
            saved.update({'youtube_video_id': video_id, 'youtube_url': url, 'privacy': privacy, 'thumbnail': thumbnail, 'bytes': size})
            _save_marker(marker, saved)
        return {'youtube_video_id': video_id, 'youtube_url': url, 'thumbnail': thumbnail if image.is_file() else None}

    def _send_video(self, video_path, body, size, session, on_progress):
        if session is None:
            session = self._start_session(body, size)
            _save_marker(marker_path(video_path), {'bytes': size, 'upload_url': session, 'thumbnail': False})
            offset = 0
        else:
            try:
                offset = self._resume_offset(session, size)
            except UploadFinished as done:
                return done.video, session
        failures = 0
        refreshed = False
        restarted = False
        with video_path.open('rb') as handle:
            while offset < size:
                if on_progress:
                    on_progress(90 + int(9 * offset / size))
                handle.seek(offset)
                chunk = handle.read(min(self.chunk_size, size - offset))
                if not chunk:
                    raise RuntimeError('영상 파일을 읽는 중 끝났습니다.')
                end = offset + len(chunk) - 1
                headers = {
                    'Authorization': self.credentials.authorization(),
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(len(chunk)),
                    'Content-Range': f'bytes {offset}-{end}/{size}',
                }
                try:
                    status, response_headers, raw = self._call('PUT', session, chunk, headers, 300)
                except OSError as error:
                    failures += 1
                    if failures > 5:
                        raise RuntimeError(f'업로드가 반복해서 끊겼습니다: {error}') from error
                    self.sleep(min(30, 2 ** failures))
                    try:
                        offset = self._resume_offset(session, size)
                    except UploadFinished as done:
                        return done.video, session
                    continue
                if status in (200, 201):
                    return _video_resource(raw), session
                if status == 308:
                    ranged = header(response_headers, 'Range')
                    new_offset = _range_end(response_headers) if ranged else end + 1
                    if new_offset <= offset:
                        failures += 1
                        if failures > 5:
                            raise RuntimeError('업로드 위치가 전진하지 않습니다.')
                    else:
                        failures = 0
                    offset = new_offset
                    if offset >= size:
                        try:
                            offset = self._resume_offset(session, size)
                        except UploadFinished as done:
                            return done.video, session
                        if offset >= size:
                            raise RuntimeError('업로드가 끝났다는 응답이 없습니다.')
                    continue
                if status == 401 and not refreshed:
                    self.credentials.force_refresh()
                    refreshed = True
                    continue
                # An expired resumable session is not a second video. Start one new session.
                if status in (404, 410) and not restarted:
                    restarted = True
                    session = self._start_session(body, size)
                    _save_marker(marker_path(video_path), {'bytes': size, 'upload_url': session, 'thumbnail': False})
                    offset = 0
                    continue
                raise RuntimeError(f'영상 업로드 실패 (HTTP {status}): {error_text(raw)}')
        raise RuntimeError('영상 업로드가 끝나기 전에 파일의 끝에 닿았습니다.')

    def _start_session(self, body, size):
        raw_body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        headers = {
            'Authorization': self.credentials.authorization(),
            'Content-Type': 'application/json; charset=UTF-8',
            'Content-Length': str(len(raw_body)),
            'X-Upload-Content-Type': 'video/mp4',
            'X-Upload-Content-Length': str(size),
            'Accept': 'application/json',
        }
        url = UPLOAD_URL + '?part=snippet,status&uploadType=resumable&notifySubscribers=false'
        status, response_headers, raw = self._call('POST', url, raw_body, headers, 60)
        if status not in (200, 201):
            raise RuntimeError(f'업로드 세션을 만들지 못했습니다 (HTTP {status}): {error_text(raw)}')
        location = header(response_headers, 'Location')
        if not location:
            raise RuntimeError('업로드 세션 주소(Location)가 없습니다.')
        return urllib.parse.urljoin(url, location)

    def _resume_offset(self, session, size):
        headers = {
            'Authorization': self.credentials.authorization(),
            'Content-Length': '0',
            'Content-Range': f'bytes */{size}',
        }
        status, response_headers, raw = self._call('PUT', session, b'', headers, 60)
        if status in (200, 201):
            raise UploadFinished(_video_resource(raw))
        if status == 308:
            return _range_end(response_headers) if header(response_headers, 'Range') else 0
        raise RuntimeError(f'업로드 재개 위치를 알지 못했습니다 (HTTP {status}): {error_text(raw)}')

    def _thumbnail(self, video_id, image):
        content_type = 'image/png' if image.suffix.lower() == '.png' else 'image/jpeg'
        data = image.read_bytes()
        headers = {
            'Authorization': self.credentials.authorization(),
            'Content-Type': content_type,
            'Content-Length': str(len(data)),
        }
        url = THUMBNAIL_URL + '?' + urllib.parse.urlencode({'videoId': video_id, 'uploadType': 'media'})
        try:
            status, _, raw = self._call('POST', url, data, headers, 60)
        except OSError as error:
            print(f'썸네일을 올리지 못했습니다 ({error}). 영상은 올라가 있습니다.', flush=True)
            return False
        if status in (200, 201):
            return True
        print(f'썸네일을 올리지 못했습니다 (HTTP {status}: {error_text(raw)}). '
              '채널이 인증되지 않았을 수 있습니다. 영상은 올라가 있습니다.', flush=True)
        return False


def _video_resource(raw):
    try:
        data = json.loads(raw.decode('utf-8'))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError('업로드 응답이 JSON이 아닙니다.') from error
    video_id = data.get('id') if isinstance(data, dict) else None
    if not video_id:
        raise RuntimeError('업로드 응답에 영상 id가 없습니다.')
    return data


def watch_url(video_id):
    return 'https://www.youtube.com/watch?v=' + urllib.parse.quote(str(video_id), safe='')


def upload_rendered(output, privacy='private', client_secret=None, token_file=None,
                    interactive=None, category=None, transport=request, on_progress=None):
    """Upload an existing render. `output` is the .mp4 path."""
    output = Path(output)
    meta_path = output.with_suffix('.youtube.json')
    if not meta_path.is_file():
        raise RuntimeError(f'유튜브 제목·설명이 없습니다: {meta_path}')
    try:
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f'유튜브 자료를 읽을 수 없습니다: {meta_path} ({error})') from error
    secret = resolve_client_secret(client_secret)
    token = resolve_token_file(token_file)
    if interactive is None:
        interactive = os.isatty(0)
    credentials = Credentials(secret, token, transport=transport)
    if not credentials.ready():
        if not interactive:
            raise RuntimeError('유튜브 갱신 토큰이 없습니다. 터미널에서 uv run scripter.py --youtube-auth 를 실행하세요.')
        authorize_interactive(secret, token, transport=transport)
    client = YouTubeClient(credentials, transport=transport)
    return client.upload(output, metadata, privacy=privacy, category=category, on_progress=on_progress)


def upload_command(output, privacy='private', client_secret=None, token_file=None):
    result = upload_rendered(output, privacy=privacy, client_secret=client_secret,
                             token_file=token_file, interactive=True)
    print(f"업로드 완료: {result['youtube_url']}", flush=True)
    if result.get('thumbnail') is False:
        print('썸네일은 올리지 못했습니다. 영상은 비공개로 올라가 있습니다. 유튜브 스튜디오에서 확인하세요.', flush=True)
    return result
