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
from unittest.mock import patch

from net import atomic_write, request

from youtube_classify import (
    classify, fit_tags, normalize_tags, resolve_category_and_tags, youtube_tag_length,
)
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
        with patch.dict(os.environ, {'YOUTUBE_CATEGORY_ID': '', 'YOUTUBE_TAGS': ''}):
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


# Realistic corrected transcripts. Category and tags are the upload contract.
SAMPLES = [
    {
        'name': 'calculus',
        'title': '정적분으로 곡선 아래 넓이 구하기',
        'description': '미적분학 강의입니다. 연속 함수의 정적분을 정의하고 넓이를 계산합니다.',
        'transcript': (
            '오늘은 정적분을 배웁니다. 미분과 적분은 역연산입니다. '
            '함수 f(x)가 연속이면 수식시작 인테그랄 영에서 일까지 수식끝 로 넓이를 구합니다. '
            '이 수학 개념은 물리학에서도 쓰입니다.'
        ),
        'categoryId': '28',
        'tags': ['정적분', '미적분학', '함수', '물리학', '미분', '수학', '적분', '곡선'],
    },
    {
        'name': 'history',
        'title': '조선 후기 실학 수업',
        'description': '고등학교 한국사 강의입니다. 실학의 개념과 대표 학자를 정리합니다.',
        'transcript': (
            '이번 시간 학생 여러분은 조선 후기의 실학을 공부합니다. '
            '시험에 자주 나오는 개념이니 교과서 필기를 확인하세요. '
            '정약용과 박지원의 주장을 비교하고 역사적 배경을 설명합니다.'
        ),
        'categoryId': '27',
        'tags': ['실학', '한국사', '고등학교', '교과서', '역사', '시험', '조선'],
    },
    {
        'name': 'vlog',
        'title': '오늘 하루 브이로그',
        'description': '카페에서 보낸 일상이랑 저녁 식사 후기입니다.',
        'transcript': (
            '안녕하세요. 오늘은 브이로그를 찍어 봤어요. '
            '아침에 동네 카페에 들렀고 일상 이야기를 남깁니다. 저녁에는 친구를 만났어요.'
        ),
        'categoryId': '22',
        'tags': ['브이로그', '일상', '하루'],
    },
    {
        'name': 'news',
        'title': '대통령 선거 개표 속보',
        'description': '국회 선거 결과와 여당 야당의 반응을 정리한 뉴스입니다.',
        'transcript': (
            '오늘 뉴스입니다. 대통령 선거 개표가 시작되었습니다. '
            '국회 앞에서 기자들이 대기하고 있고 여당과 야당이 각각 입장을 냈습니다. '
            '정부의 정책 발표도 예정되어 있습니다.'
        ),
        'categoryId': '25',
        'tags': ['대통령', '선거', '속보', '국회', '뉴스', '야당', '여당', '기자'],
    },
    {
        'name': 'programming',
        'title': '파이썬으로 구현하는 정렬 알고리즘',
        'description': '퀵 정렬과 병합 정렬의 시간 복잡도를 비교하고 Python 코드를 작성합니다.',
        'transcript': (
            '오늘은 파이썬으로 정렬 알고리즘을 구현합니다. '
            '퀵 정렬의 평균 시간 복잡도는 n log n 입니다. '
            'Python으로 코드를 같이 작성하고 자료구조를 확인합니다.'
        ),
        'categoryId': '28',
        'tags': ['알고리즘', '파이썬', '정렬', 'Python', '병합 정렬', '시간 복잡도', '퀵 정렬', '코드'],
    },
    {
        'name': 'cooking',
        'title': '집에서 만드는 김치찌개 레시피',
        'description': '김치찌개 요리 방법을 단계별로 따라 합니다.',
        'transcript': '오늘은 김치찌개 만드는 법을 알려드릴게요. 레시피대로 돼지고기를 볶고 김치를 넣습니다. 요리 초보도 따라 할 수 있습니다.',
        'categoryId': '26',
        'tags': ['레시피', '요리', '만드는 법', '김치찌개'],
    },
    {
        'name': 'music',
        'title': '새 앨범 녹음 비하인드',
        'description': '이번 싱글의 멜로디와 가사를 스튜디오에서 녹음했습니다.',
        'transcript': '새 노래의 앨범 작업입니다. 피아노로 멜로디를 잡고 가사를 녹음했습니다. 콘서트 전에 편곡을 마칩니다.',
        'categoryId': '10',
        'tags': ['앨범', '멜로디', '가사', '콘서트', '노래', '피아노'],
    },
    {
        'name': 'film',
        'title': '단편 영화 제작 과정',
        'description': '촬영과 시나리오, 애니메이션 연출을 기록했습니다.',
        'transcript': '이번 단편 영화는 촬영 일정이 짧았습니다. 시나리오를 고치고 애니메이션 느낌을 살렸습니다.',
        'categoryId': '1',
        'tags': ['단편 영화', '애니메이션', '시나리오', '촬영', '영화'],
    },
    {
        'name': 'autos',
        'title': '전기차 시승',
        'description': '자동차 주행과 차량 기능을 살펴봅니다.',
        'transcript': '오늘 전기차를 시승했습니다. 자동차의 주행 감각과 차량 옵션을 확인합니다.',
        'categoryId': '2',
        'tags': ['시승', '전기차', '자동차', '차량'],
    },
    {
        'name': 'pets',
        'title': '우리 강아지 산책',
        'description': '고양이랑 같이 사는 반려동물 이야기입니다.',
        'transcript': '강아지 산책을 다녀왔고 고양이도 창가에 앉아 있었습니다. 반려동물 둘을 돌보는 하루입니다.',
        'categoryId': '15',
        'tags': ['강아지', '고양이', '반려동물', '산책'],
    },
    {
        'name': 'sports',
        'title': '챔피언스리그 축구 하이라이트',
        'description': '선수들의 골 장면과 경기 결과입니다.',
        'transcript': '오늘 축구 하이라이트입니다. 선수가 추가 시간에 골을 넣었습니다.',
        'categoryId': '17',
        'tags': ['축구', '하이라이트', '선수', '챔피언스리그'],
    },
    {
        'name': 'travel',
        'title': '도쿄 여행 코스',
        'description': '관광 명소와 호텔을 걷습니다.',
        'transcript': '도쿄 여행 이틀째입니다. 관광지를 걷고 호텔로 돌아왔습니다.',
        'categoryId': '19',
        'tags': ['여행', '관광', '호텔', '도쿄', '코스'],
    },
    {
        'name': 'comedy',
        'title': '개그 콩트 모음',
        'description': '코미디 유머 영상입니다.',
        'transcript': '웃긴 개그 콩트를 모았습니다. 코미디 유머가 이어집니다.',
        'categoryId': '23',
        'tags': ['개그', '콩트', '코미디', '유머', '웃긴'],
    },
    {
        'name': 'entertainment',
        'title': '아이돌 예능 리뷰',
        'description': '연예인 인터뷰와 드라마 이야기를 모았습니다.',
        'transcript': '이번 예능에 나온 아이돌과 드라마 촬영 비하인드를 이야기합니다.',
        'categoryId': '24',
        'tags': ['아이돌', '예능', '드라마', '연예인', '리뷰'],
    },
    {
        'name': 'nonprofit',
        'title': '겨울 기부 캠페인 안내',
        'description': '비영리 단체의 봉사와 모금 소식입니다.',
        'transcript': '겨울 기부 캠페인에 함께해 주세요. 비영리 단체가 모금한 금액은 봉사 활동에 씁니다.',
        'categoryId': '29',
        'tags': ['기부', '비영리', '모금', '봉사', '캠페인'],
    },
    {
        'name': 'gaming',
        'title': '엘든링 보스 공략',
        'description': '게임 플레이 영상입니다.',
        'transcript': '이번 보스 공략입니다. 게임 플레이 중에 패턴을 익히면 쓰러뜨릴 수 있습니다.',
        'categoryId': '20',
        'tags': ['공략', '보스', '게임', '플레이', '엘든링'],
    },
]


def _limits(testcase, tags):
    testcase.assertLessEqual(len(tags), 8)
    testcase.assertLess(youtube_tag_length(tags), 500)
    testcase.assertTrue(all(2 <= len(tag) <= 30 for tag in tags))
    testcase.assertNotIn('수식시작', tags)
    testcase.assertNotIn('수식끝', tags)


class ClassifyTests(unittest.TestCase):
    def test_samples_fill_one_real_category_and_a_handful_of_tags(self):
        seen = set()
        for sample in SAMPLES:
            result = classify(sample['title'], sample['description'], sample['transcript'])
            self.assertEqual(result['categoryId'], sample['categoryId'], sample['name'])
            self.assertIn(result['categoryId'], {
                '1', '2', '10', '15', '17', '19', '20', '22', '23', '24', '25', '26', '27', '28', '29',
            })
            self.assertEqual(result['tags'], sample['tags'], sample['name'])
            _limits(self, result['tags'])
            seen.add(result['categoryId'])
        for required in ('27', '28', '22'):
            self.assertIn(required, seen)

    def test_programming_tags_mix_korean_and_english(self):
        sample = next(item for item in SAMPLES if item['name'] == 'programming')
        tags = classify(sample['title'], sample['description'], sample['transcript'])['tags']
        self.assertIn('파이썬', tags)
        self.assertIn('Python', tags)
        self.assertIn('알고리즘', tags)
        self.assertGreater(youtube_tag_length(tags), sum(len(tag) for tag in tags))

    def test_formula_marker_and_product_name_are_not_tags(self):
        calculus = next(item for item in SAMPLES if item['name'] == 'calculus')
        self.assertIn('수식시작', calculus['transcript'])
        self.assertNotIn('수식시작', classify(calculus['title'], calculus['description'], calculus['transcript'])['tags'])
        result = classify('마크벡터 소개', '마크벡터로 남긴 기록입니다.', '오늘은 마크벡터를 열었습니다.')
        self.assertNotIn('벡터', result['tags'])
        self.assertEqual(result['categoryId'], '27')
        self.assertEqual(result['categorySource'], 'default')

    def test_weak_text_stays_education_without_invented_tags(self):
        result = classify(
            '오늘의 정리', '앞에서 다룬 내용을 다시 봅니다.',
            '네 지금까지 이야기한 내용을 짧게 정리합니다. 질문은 다음에 받겠습니다.')
        self.assertEqual(result['categoryId'], '27')
        self.assertEqual(result['categorySource'], 'default')
        self.assertEqual(result['tags'], [])

    def test_chapter_titles_count_as_description(self):
        resolved = resolve_category_and_tags({
            'title': '정리', 'description': '설명',
            'chapters': [
                {'start': '00:00', 'title': '정적분의 정의'},
                {'start': '01:00', 'title': '미분과의 관계'},
                {'start': '02:00', 'title': '물리학에 쓰는 적분'},
            ],
        }, environ={})
        self.assertEqual(resolved['categoryId'], '28')
        self.assertIn('정적분', resolved['tags'])

    def test_argument_beats_file_and_file_beats_environment(self):
        forced = resolve_category_and_tags(
            {'title': '정적분으로 넓이 구하기', 'description': '미적분학 강의',
             'categoryId': '22', 'tags': ['파일태그']},
            transcript='미분과 적분을 배웁니다.',
            category='28', tags='수학, 적분',
            environ={'YOUTUBE_CATEGORY_ID': '25', 'YOUTUBE_TAGS': '뉴스'},
        )
        self.assertEqual(forced['categoryId'], '28')
        self.assertEqual(forced['categorySource'], 'argument')
        self.assertEqual(forced['tags'], ['수학', '적분'])
        self.assertEqual(forced['tagsSource'], 'argument')

        from_file = resolve_category_and_tags(
            {'title': '오늘 하루 브이로그', 'description': '일상', 'categoryId': 28},
            environ={'YOUTUBE_CATEGORY_ID': '22'},
        )
        self.assertEqual(from_file['categoryId'], '28')
        self.assertEqual(from_file['categorySource'], 'file')
        self.assertEqual(from_file['category'], 'Science & Technology')

        from_env = resolve_category_and_tags(
            {'title': '정적분으로 넓이 구하기', 'description': '미적분학'},
            transcript='미분과 적분',
            environ={'YOUTUBE_CATEGORY_ID': '22', 'YOUTUBE_TAGS': '직접,지정'},
        )
        self.assertEqual(from_env['categoryId'], '22')
        self.assertEqual(from_env['categorySource'], 'env')
        self.assertEqual(from_env['tags'], ['직접', '지정'])
        self.assertEqual(from_env['tagsSource'], 'env')

    def test_empty_file_tags_mean_no_tags(self):
        resolved = resolve_category_and_tags(
            {'title': '정적분으로 넓이 구하기', 'description': '수학 강의', 'tags': []},
            environ={})
        self.assertEqual(resolved['categoryId'], '28')
        self.assertEqual(resolved['tags'], [])
        self.assertEqual(resolved['tagsSource'], 'file')

    def test_invalid_override_is_rejected(self):
        with self.assertRaises(RuntimeError):
            resolve_category_and_tags({'title': '강의', 'categoryId': '99'}, environ={})
        with self.assertRaises(RuntimeError):
            resolve_category_and_tags({'title': '강의', 'categoryId': 'Education'}, environ={})
        with self.assertRaises(RuntimeError):
            normalize_tags('@@@', required=True)
        self.assertEqual(normalize_tags(''), [])
        self.assertEqual(normalize_tags('수학, 적분, calculus'), ['수학', '적분', 'calculus'])

    def test_tag_budget_matches_youtube_rules(self):
        self.assertEqual(youtube_tag_length(['Foo-Baz']), 7)
        self.assertEqual(youtube_tag_length(['Foo Baz']), 9)
        self.assertEqual(youtube_tag_length(['ab', 'cd']), 5)
        self.assertEqual(fit_tags(['가' * 31, '수학', '수학']), ['수학'])
        many = fit_tags(['가' * 20 + f'{index:02d}' for index in range(40)], limit=30)
        self.assertLess(youtube_tag_length(many), 500)
        self.assertTrue(all(len(tag) <= 30 for tag in many))
        self.assertGreater(len(many), 8)

    def test_video_body_sends_category_and_tags(self):
        sample = next(item for item in SAMPLES if item['name'] == 'programming')
        with patch.dict(os.environ, {'YOUTUBE_CATEGORY_ID': '', 'YOUTUBE_TAGS': ''}):
            body = video_body(
                {'title': sample['title'], 'description': sample['description']},
                transcript=sample['transcript'])
        self.assertEqual(body['snippet']['categoryId'], '28')
        self.assertEqual(body['snippet']['tags'], sample['tags'])
        self.assertEqual(body['status']['privacyStatus'], 'private')
        self.assertEqual(body['snippet']['defaultLanguage'], 'ko')
        with patch.dict(os.environ, {'YOUTUBE_CATEGORY_ID': '', 'YOUTUBE_TAGS': ''}):
            empty = video_body(
                {'title': '오늘의 정리', 'description': '앞에서 다룬 내용을 다시 봅니다.'},
                transcript='네 지금까지 이야기한 내용을 짧게 정리합니다.')
        self.assertEqual(empty['snippet']['categoryId'], '27')
        self.assertNotIn('tags', empty['snippet'])


class ClassificationUploadTests(unittest.TestCase):
    def test_insert_body_uses_the_corrected_transcript(self):
        sample = next(item for item in SAMPLES if item['name'] == 'calculus')
        calls = []

        def transport(method, url, data=None, headers=None, timeout=60):
            calls.append((method, url, data))
            if method == 'POST' and 'uploadType=resumable' in url:
                posted = json.loads(data.decode('utf-8'))
                self.assertEqual(posted['snippet']['categoryId'], '28')
                self.assertEqual(posted['snippet']['tags'], sample['tags'])
                self.assertNotIn('수식시작', posted['snippet']['tags'])
                return 200, {'Location': 'https://upload.example/session'}, b''
            if method == 'PUT':
                return 200, {}, b'{"id":"calc01"}'
            raise AssertionError((method, url))

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / 'video.mp4'
            video.write_bytes(b'0123456789')
            meta = {'title': sample['title'], 'description': sample['description']}
            video.with_suffix('.youtube.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
            video.with_suffix('.txt').write_text(sample['transcript'], encoding='utf-8')
            credentials = Credentials(client_json(folder), token_json(folder, time.time() + 600), transport=transport)
            client = YouTubeClient(credentials, transport=transport, sleep=lambda seconds: None)
            with patch.dict(os.environ, {'YOUTUBE_CATEGORY_ID': '', 'YOUTUBE_TAGS': ''}):
                result = client.upload(video, meta)
            self.assertEqual(result['categoryId'], '28')
            self.assertEqual(result['tags'], sample['tags'])
            self.assertEqual(json.loads(video.with_suffix('.youtube-upload.json').read_text())['categoryId'], '28')
            self.assertEqual(video.with_suffix('.youtube.json').read_text(encoding='utf-8'),
                             json.dumps(meta, ensure_ascii=False))
            self.assertTrue(all(method == 'PUT' or 'uploadType=resumable' in url for method, url, _data in calls))
            self.assertFalse(any('videos.update' in url or '/videos/' in url for _method, url, _data in calls))

    def test_generic_title_still_follows_the_transcript(self):
        sample = next(item for item in SAMPLES if item['name'] == 'calculus')
        with patch.dict(os.environ, {'YOUTUBE_CATEGORY_ID': '', 'YOUTUBE_TAGS': ''}):
            plain = video_body({'title': '강의 제목', 'description': '설명입니다.'})
            informed = video_body(
                {'title': '강의 제목', 'description': '설명입니다.'},
                transcript=sample['transcript'])
        self.assertEqual(plain['snippet']['categoryId'], '27')
        self.assertEqual(informed['snippet']['categoryId'], '28')
        self.assertIn('정적분', informed['snippet']['tags'])

    def test_existing_video_is_not_modified(self):
        def transport(*args, **kwargs):
            raise AssertionError('already uploaded video must not be changed')

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / 'video.mp4'
            video.write_bytes(b'abcd')
            meta = {'title': '정적분으로 넓이 구하기', 'description': '미적분학 강의'}
            original = json.dumps(meta, ensure_ascii=False)
            video.with_suffix('.youtube.json').write_text(original, encoding='utf-8')
            video.with_suffix('.txt').write_text('미분과 적분을 계속 설명합니다.', encoding='utf-8')
            marker = {'youtube_video_id': 'vid123', 'thumbnail': True, 'bytes': 4}
            video.with_suffix('.youtube-upload.json').write_text(json.dumps(marker), encoding='utf-8')
            before = video.with_suffix('.youtube-upload.json').read_text(encoding='utf-8')
            credentials = Credentials(client_json(folder), token_json(folder, time.time() + 600), transport=transport)
            client = YouTubeClient(credentials, transport=transport, sleep=lambda seconds: None)
            result = client.upload(video, meta)
            self.assertEqual(result['youtube_video_id'], 'vid123')
            self.assertNotIn('categoryId', result)
            self.assertEqual(video.with_suffix('.youtube-upload.json').read_text(encoding='utf-8'), before)
            self.assertEqual(video.with_suffix('.youtube.json').read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
