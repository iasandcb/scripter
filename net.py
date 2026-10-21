"""HTTP without following 308. Resumable YouTube uploads use 308 as success."""
import http.client
import json
import os
import urllib.parse
from contextlib import contextmanager
from pathlib import Path


USER_AGENT = 'scripter'


class HTTPError(Exception):
    def __init__(self, status, body, headers=None):
        self.status = status
        self.body = body
        self.headers = headers
        super().__init__(f'HTTP {status}: {error_text(body)}')


def header(headers, name):
    """Case-insensitive header lookup. `headers` is an HTTP message or a dict."""
    if not headers:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return value
    return None


def error_text(body, limit=500):
    """A short message from a JSON or text error body. Tokens are not added."""
    if body is None:
        return ''
    if isinstance(body, bytes):
        text = body.decode('utf-8', 'replace')
    else:
        text = str(body)
    text = text.strip()
    try:
        data = json.loads(text)
    except ValueError:
        return text[:limit]
    if isinstance(data, dict):
        detail = data.get('detail', data.get('error', data.get('error_description')))
        if isinstance(detail, dict):
            detail = detail.get('message') or detail
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    parts.append(str(item.get('msg', item)))
                else:
                    parts.append(str(item))
            detail = '; '.join(parts)
        if detail:
            return str(detail)[:limit]
    return text[:limit]


def decode_json(body):
    if isinstance(body, bytes):
        body = body.decode('utf-8')
    if not body:
        return {}
    return json.loads(body)


@contextmanager
def open_request(method, url, data=None, headers=None, timeout=60):
    """One request. The caller reads the response before the context exits."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError(f'지원하지 않는 주소입니다: {url}')
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    connection_cls = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
    connection = connection_cls(parts.hostname, port, timeout=timeout)
    path = parts.path or '/'
    if parts.query:
        path += '?' + parts.query
    sent = {'User-Agent': USER_AGENT}
    if headers:
        sent.update({str(key): str(value) for key, value in headers.items()})
    try:
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        try:
            yield response
        finally:
            response.close()
    finally:
        connection.close()


def request(method, url, data=None, headers=None, timeout=60):
    """Return (status, headers, body). Does not follow redirects."""
    with open_request(method, url, data=data, headers=headers, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def atomic_write(path, text, mode=0o600):
    """Write `text` and fsync, then replace `path`. Mode stays 600 after umask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError(f'심볼릭 링크에는 비밀을 쓰지 않습니다: {path}')
    temporary = path.with_name(path.name + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    os.chmod(path, mode)


def retry_after(headers):
    value = header(headers, 'Retry-After')
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds < 0 or seconds > 300:
        return None
    return seconds
