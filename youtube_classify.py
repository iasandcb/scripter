"""Pick a YouTube category and tags from a video's own text.

This runs offline at upload time. Title, description, and the corrected
transcript are scored against the assignable YouTube categoryIds. No network
and no API key: the upload already has the text, and a second model call would
only add cost and a failure point.

Manual overrides win, per field, in this order: an explicit argument (the
CLI flags), a value written in `.youtube.json`, then `YOUTUBE_CATEGORY_ID` /
`YOUTUBE_TAGS`. Anything left unset is filled from the text. An empty score
falls back to Education (27), which is what uploads used before classification.

YouTube counts tag length as characters. A comma between tags counts, and a
tag that contains a space counts two extra characters because the API quotes
it. The total must stay under 500. Each tag is also kept to 30 characters.
"""
import functools
import os
import re


# Assignable video categoryIds. Names are the Data API's English titles.
# https://developers.google.com/youtube/v3/docs/videoCategories/list
CATEGORIES = {
    '1': 'Film & Animation',
    '2': 'Autos & Vehicles',
    '10': 'Music',
    '15': 'Pets & Animals',
    '17': 'Sports',
    '19': 'Travel & Events',
    '20': 'Gaming',
    '22': 'People & Blogs',
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Howto & Style',
    '27': 'Education',
    '28': 'Science & Technology',
    '29': 'Nonprofits & Activism',
}
DEFAULT_CATEGORY = '27'
# One modest title hit (weight 3 × 3) clears this. A single passing mention
# in the transcript does not, so a lecture that names another topic once
# stays with the stronger subject.
MIN_SCORE = 8
# Higher wins a tied score. Specific topics beat the two catch-all buckets.
TIE_BREAK = {
    '28': 90, '26': 80, '25': 78, '20': 76, '17': 74, '10': 72, '1': 70,
    '15': 68, '2': 66, '19': 64, '23': 62, '29': 60, '24': 50, '27': 20, '22': 10,
}
MAX_TAGS = 8
MAX_TAG_LENGTH = 30
# YouTube's ceiling is 500. Stay strictly under it.
MAX_TAG_TOTAL = 499
MAX_FIELD_CHARS = 80000

# phrase, weight, tag. tag None still scores the category but is not emitted.
# Longer phrases are matched first and consume their span, so 적분 does not
# also fire inside 정적분.
KEYWORDS = {
    '28': (
        ('machine learning', 6, 'machine learning'),
        ('deep learning', 6, 'deep learning'),
        ('neural network', 5, 'neural network'),
        ('인공지능', 5, '인공지능'),
        ('머신러닝', 5, '머신러닝'),
        ('딥러닝', 5, '딥러닝'),
        ('신경망', 5, '신경망'),
        ('알고리즘', 5, '알고리즘'),
        ('자료구조', 5, '자료구조'),
        ('프로그래밍', 5, '프로그래밍'),
        ('소프트웨어', 4, '소프트웨어'),
        ('시간 복잡도', 4, '시간 복잡도'),
        ('시간복잡도', 4, '시간복잡도'),
        ('양자역학', 6, '양자역학'),
        ('미적분학', 5, '미적분학'),
        ('미분방정식', 5, '미분방정식'),
        ('선형대수', 5, '선형대수'),
        ('정적분', 5, '정적분'),
        ('미적분', 5, '미적분'),
        ('도함수', 4, '도함수'),
        ('물리학', 5, '물리학'),
        ('생물학', 5, '생물학'),
        ('자바스크립트', 5, '자바스크립트'),
        ('javascript', 5, 'JavaScript'),
        ('mathematics', 5, 'mathematics'),
        ('algorithm', 5, 'algorithm'),
        ('calculus', 5, 'calculus'),
        ('physics', 5, 'physics'),
        ('quantum', 4, 'quantum'),
        ('chemistry', 4, 'chemistry'),
        ('biology', 4, 'biology'),
        ('software', 4, 'software'),
        ('파이썬', 5, '파이썬'),
        ('python', 5, 'Python'),
        ('컴퓨터', 3, '컴퓨터'),
        ('코딩', 4, '코딩'),
        ('수학', 4, '수학'),
        ('화학', 4, '화학'),
        ('과학', 4, '과학'),
        ('공학', 4, '공학'),
        ('미분', 4, '미분'),
        ('적분', 4, '적분'),
        ('함수', 3, '함수'),
        ('방정식', 3, '방정식'),
        ('극한', 3, '극한'),
        ('행렬', 3, '행렬'),
        ('벡터', 3, '벡터'),
        ('확률', 3, '확률'),
        ('통계', 3, '통계'),
        ('기하', 3, '기하'),
        ('데이터', 3, '데이터'),
        ('코드', 3, '코드'),
        ('정렬', 3, '정렬'),
        ('퀵 정렬', 4, '퀵 정렬'),
        ('병합 정렬', 4, '병합 정렬'),
        ('technology', 3, 'technology'),
        ('engineering', 3, 'engineering'),
    ),
    '27': (
        ('고등학교', 4, '고등학교'),
        ('중학교', 4, '중학교'),
        ('초등학교', 4, '초등학교'),
        ('한국사', 5, '한국사'),
        ('세계사', 5, '세계사'),
        ('교과서', 4, '교과서'),
        ('경제학', 4, '경제학'),
        ('실학', 4, '실학'),
        ('대학교', 3, '대학교'),
        ('문법', 4, '문법'),
        ('문학', 4, '문학'),
        ('철학', 4, '철학'),
        ('역사', 4, '역사'),
        ('시험', 3, '시험'),
        ('교육', 4, '교육'),
        ('강의', 4, None),
        ('수업', 4, None),
        ('공부', 3, None),
        ('학습', 3, None),
        ('학생', 2, None),
        ('개념', 2, None),
        ('lecture', 4, 'lecture'),
        ('lesson', 3, 'lesson'),
        ('textbook', 4, 'textbook'),
        ('education', 4, 'education'),
        ('university', 3, 'university'),
    ),
    '22': (
        ('브이로그', 6, '브이로그'),
        ('vlog', 6, 'vlog'),
        ('daily life', 5, 'daily life'),
        ('lifestyle', 4, 'lifestyle'),
        ('블로그', 4, '블로그'),
        ('데일리', 4, '데일리'),
        ('일상', 4, '일상'),
        ('일기', 3, '일기'),
    ),
    '26': (
        ('만드는 법', 6, '만드는 법'),
        ('how to', 5, 'how to'),
        ('레시피', 6, '레시피'),
        ('recipe', 6, 'recipe'),
        ('메이크업', 5, '메이크업'),
        ('makeup', 5, 'makeup'),
        ('따라하기', 4, '따라하기'),
        ('튜토리얼', 4, '튜토리얼'),
        ('tutorial', 3, 'tutorial'),
        ('하는 법', 4, '하는 법'),
        ('패션', 4, '패션'),
        ('요리', 4, '요리'),
        ('diy', 4, 'DIY'),
    ),
    '25': (
        ('breaking news', 6, 'breaking news'),
        ('기자회견', 4, '기자회견'),
        ('대통령', 5, '대통령'),
        ('election', 5, 'election'),
        ('politics', 4, 'politics'),
        ('선거', 5, '선거'),
        ('속보', 5, '속보'),
        ('국회', 4, '국회'),
        ('여당', 4, '여당'),
        ('야당', 4, '야당'),
        ('정치', 4, '정치'),
        ('뉴스', 4, '뉴스'),
        ('정부', 3, '정부'),
        ('정책', 3, '정책'),
        ('기자', 3, '기자'),
    ),
    '10': (
        ('concert', 5, 'concert'),
        ('album', 5, 'album'),
        ('lyrics', 3, 'lyrics'),
        ('music', 4, 'music'),
        ('콘서트', 5, '콘서트'),
        ('멜로디', 4, '멜로디'),
        ('피아노', 4, '피아노'),
        ('앨범', 5, '앨범'),
        ('음악', 4, '음악'),
        ('연주', 4, '연주'),
        ('노래', 4, '노래'),
        ('가사', 3, '가사'),
    ),
    '20': (
        ('gameplay', 6, 'gameplay'),
        ('게임플레이', 6, '게임플레이'),
        ('esports', 5, 'esports'),
        ('이스포츠', 5, '이스포츠'),
        ('gaming', 5, 'gaming'),
        ('공략', 5, '공략'),
        ('스테이지', 3, '스테이지'),
        ('플레이', 3, '플레이'),
        ('보스', 3, '보스'),
        ('랭크', 3, '랭크'),
        ('게임', 3, '게임'),
    ),
    '17': (
        ('basketball', 5, 'basketball'),
        ('baseball', 5, 'baseball'),
        ('olympics', 5, 'olympics'),
        ('soccer', 5, 'soccer'),
        ('sports', 4, 'sports'),
        ('올림픽', 5, '올림픽'),
        ('하이라이트', 3, '하이라이트'),
        ('축구', 5, '축구'),
        ('야구', 5, '야구'),
        ('농구', 5, '농구'),
        ('선수', 3, '선수'),
    ),
    '19': (
        ('tourist', 4, 'tourist'),
        ('travel', 5, 'travel'),
        ('hotel', 4, 'hotel'),
        ('관광', 4, '관광'),
        ('여행', 5, '여행'),
        ('호텔', 4, '호텔'),
        ('맛집', 3, '맛집'),
        ('축제', 4, '축제'),
    ),
    '1': (
        ('animation', 5, 'animation'),
        ('단편 영화', 6, '단편 영화'),
        ('애니메이션', 5, '애니메이션'),
        ('시나리오', 4, '시나리오'),
        ('cinema', 4, 'cinema'),
        ('영화', 5, '영화'),
        ('촬영', 3, '촬영'),
        ('film', 3, 'film'),
    ),
    '15': (
        ('반려동물', 5, '반려동물'),
        ('강아지', 5, '강아지'),
        ('고양이', 5, '고양이'),
        ('kitten', 5, 'kitten'),
        ('puppy', 5, 'puppy'),
        ('dog', 3, 'dog'),
        ('cat', 3, 'cat'),
        ('펫', 3, '펫'),
    ),
    '2': (
        ('electric vehicle', 5, 'electric vehicle'),
        ('automotive', 4, 'automotive'),
        ('전기차', 5, '전기차'),
        ('자동차', 5, '자동차'),
        ('시승', 5, '시승'),
        ('차량', 4, '차량'),
        ('운전', 3, '운전'),
        ('car', 3, 'car'),
    ),
    '23': (
        ('stand-up', 5, 'stand-up'),
        ('comedy', 5, 'comedy'),
        ('코미디', 5, '코미디'),
        ('콩트', 5, '콩트'),
        ('개그', 5, '개그'),
        ('유머', 4, '유머'),
        ('웃긴', 3, '웃긴'),
    ),
    '24': (
        ('entertainment', 4, 'entertainment'),
        ('celebrity', 4, 'celebrity'),
        ('연예인', 4, '연예인'),
        ('아이돌', 5, '아이돌'),
        ('드라마', 4, '드라마'),
        ('예능', 5, '예능'),
        ('연예', 3, '연예'),
    ),
    '29': (
        ('nonprofit', 5, 'nonprofit'),
        ('donation', 4, 'donation'),
        ('charity', 5, 'charity'),
        ('비영리', 5, '비영리'),
        ('인권', 4, '인권'),
        ('모금', 4, '모금'),
        ('봉사', 4, '봉사'),
        ('기부', 5, '기부'),
    ),
}

STOPWORDS = {
    '그리고', '그래서', '그러나', '하지만', '그러면', '그런데', '그러니까', '왜냐하면',
    '이제', '오늘', '오늘의', '이번', '우리', '여러분', '여기', '거기', '무엇', '이건', '그건',
    '하는', '하다', '했다', '합니다', '입니다', '있다', '있습니다', '없다', '없습니다',
    '된다', '됩니다', '위해', '위한', '대한', '통해', '대해서', '에서', '으로',
    '처럼', '같이', '정도', '부분', '경우', '생각', '이야기', '얘기', '정말', '아주',
    '매우', '그냥', '일단', '다음', '먼저', '바로', '이런', '그런', '저런',
    '안녕', '안녕하세요', '네', '예', '응', '음', '어', '자', '그럼', '때문에',
    '수식', '수식시작', '수식끝', '구하기', '알려', '드립니다', '해보세요',
    '이것', '저것', '그것', '무슨', '어떤', '어떻게', '왜',
    '강의', '수업', '제목', '설명', '정리', '내용', '질문', '영상', '소개', '안내',
    '과정', '시작', '본론', '아래', '후기', '전기', '녹음', '비하인드', '모음', '제작',
    '겨울', '봄', '여름', '가을',
    '만드는', '하는', '따라', '구현', '작성', '기록', '비교', '장면', '결과',
    'the', 'and', 'for', 'with', 'this', 'that', 'from', 'you', 'are', 'was',
    'were', 'have', 'has', 'not', 'but', 'can', 'will', 'just', 'about', 'into',
    'your', 'our', 'its', 'they', 'them', 'his', 'her', 'she', 'him', 'how',
}
_PARTICLES = tuple(sorted((
    '에서', '으로', '에게', '한테', '처럼', '까지', '부터', '보다', '이나',
    '은', '는', '이', '가', '을', '를', '의', '에', '로', '과', '와', '도', '만',
), key=len, reverse=True))
_TOKEN = re.compile(r'[가-힣]{2,}|[A-Za-z][A-Za-z0-9+#]{2,}')
_FORBIDDEN_TAG = re.compile(r'[<>"\n\r,]')


def category_id(value):
    """Validate an assignable YouTube categoryId. Raises RuntimeError otherwise."""
    chosen = str(value).strip()
    if not chosen.isdigit() or chosen not in CATEGORIES:
        examples = '27 Education, 28 Science & Technology, 22 People & Blogs'
        raise RuntimeError(
            f'유튜브 categoryId가 올바르지 않습니다: {value}. 예: {examples}.')
    return chosen


def youtube_tag_length(tags):
    """Characters YouTube counts: commas between tags, plus quotes around spaces."""
    if not tags:
        return 0
    parts = [f'"{tag}"' if ' ' in tag else tag for tag in tags]
    return len(','.join(parts))


def fit_tags(tags, limit=MAX_TAGS, max_length=MAX_TAG_LENGTH, max_total=MAX_TAG_TOTAL):
    """Drop blanks, duplicates, and anything that would break YouTube's tag limits.

    Order is kept. The first tag that would push the total to 500 or more ends
    the list, so the earlier tags (the more specific ones) survive.
    """
    fitted = []
    seen = set()
    total = 0
    for raw in tags:
        tag = ' '.join(str(raw).split())
        if not tag or len(tag) < 2 or len(tag) > max_length:
            continue
        if _FORBIDDEN_TAG.search(tag) or not re.search(r'[가-힣A-Za-z0-9]', tag):
            continue
        key = tag.casefold()
        if key in seen:
            continue
        units = len(tag) + (2 if ' ' in tag else 0)
        extra = units if not fitted else units + 1
        if total + extra > max_total:
            break
        seen.add(key)
        fitted.append(tag)
        total += extra
        if len(fitted) >= limit:
            break
    return fitted


def normalize_tags(value, limit=30, required=False):
    """Parse a manual tag override (a list, or a comma-separated string)."""
    if isinstance(value, str):
        if not value.strip():
            return []
        parts = [part.strip() for part in value.split(',')]
    elif isinstance(value, (list, tuple)):
        if not value:
            return []
        parts = list(value)
    else:
        raise RuntimeError('태그는 문자열 목록이거나 쉼표로 구분된 문자열이어야 합니다.')
    fitted = fit_tags(parts, limit=limit)
    if required and not fitted:
        raise RuntimeError(
            '쓸 수 있는 태그가 없습니다. 한 태그는 2자 이상 30자 이하이고, '
            '<>와 따옴표, 쉼표는 넣을 수 없으며, 합계는 500자 미만이어야 합니다.')
    return fitted


@functools.lru_cache(maxsize=512)
def _pattern(phrase):
    lowered = phrase.lower()
    escaped = re.escape(lowered)
    if re.search(r'[가-힣]', lowered):
        # Korean stems keep a following particle (적분을) but do not match
        # inside a longer word (정적분, 마크벡터).
        return re.compile('(?<![가-힣])' + escaped)
    return re.compile(r'(?<![a-z0-9])' + escaped + r'(?![a-z0-9])')


def _clip(text):
    text = '' if text is None else str(text)
    if len(text) > MAX_FIELD_CHARS:
        return text[:MAX_FIELD_CHARS]
    return text


def _match(text, keywords):
    """Return (phrase, weight, tag, counted hits). Longer phrases consume their span."""
    if not text:
        return []
    lowered = text.lower()
    occupied = bytearray(len(lowered))
    found = []
    ordered = sorted(keywords, key=lambda item: len(item[0]), reverse=True)
    for phrase, weight, tag in ordered:
        hits = 0
        for match in _pattern(phrase).finditer(lowered):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            occupied[start:end] = b'\x01' * (end - start)
            hits += 1
        if hits:
            found.append((phrase, weight, tag, min(hits, 2)))
    return found


def score_categories(title, description='', transcript=''):
    """CategoryId → score. Title counts 3, description 2, transcript 1."""
    scores = {key: 0 for key in CATEGORIES}
    fields = ((_clip(title), 3), (_clip(description), 2), (_clip(transcript), 1))
    for text, multiplier in fields:
        if not text:
            continue
        for category, keywords in KEYWORDS.items():
            hits = _match(text, keywords)
            if hits:
                scores[category] += multiplier * sum(weight * count for _, weight, _, count in hits)
    return scores


def _content_tokens(text):
    tokens = []
    for raw in _TOKEN.findall(text or ''):
        token = _normalize_token(raw)
        if token:
            tokens.append(token)
    return tokens


def _normalize_token(token):
    if re.fullmatch(r'[A-Za-z][A-Za-z0-9+#]{2,}', token):
        if token.casefold() in STOPWORDS or len(token) > MAX_TAG_LENGTH:
            return None
        return token
    if not re.fullmatch(r'[가-힣]{2,}', token):
        return None
    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) > len(particle):
            stem = token[:-len(particle)]
            if len(stem) < 2:
                return None
            token = stem
            break
    if token.casefold() in STOPWORDS or len(token) > MAX_TAG_LENGTH:
        return None
    return token


def _covered(token, chosen):
    key = token.casefold()
    for tag in chosen:
        other = tag.casefold()
        if key == other or key.startswith(other) or other.startswith(key):
            return True
    return False


def _appearances(title, description, transcript):
    """tag → (field rank, weight, category). A title hit outranks a later one."""
    found = {}
    fields = ((_clip(title), 0), (_clip(description), 1), (_clip(transcript), 2))
    for text, rank in fields:
        if not text:
            continue
        for category, keywords in KEYWORDS.items():
            for phrase, weight, tag, _count in _match(text, keywords):
                if not tag:
                    continue
                current = found.get(tag)
                if current is None or rank < current[0] or (rank == current[0] and weight > current[1]):
                    found[tag] = (rank, weight, category)
    return found


def _frequent(transcript, limit=MAX_TAGS):
    counts = {}
    for token in _content_tokens(transcript):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
    return [token for token in ranked if counts[token] >= 2][:limit]


def suggest_tags(title, description='', transcript='', category=None):
    """A handful of tags that actually occur in the text, best signal first."""
    appearances = _appearances(title, description, transcript)
    own = [(rank, weight, tag) for tag, (rank, weight, cid) in appearances.items() if cid == category]
    own.sort(key=lambda item: (item[0], -item[1], item[2]))
    tags = [tag for _rank, _weight, tag in own]
    if len(tags) < 3:
        others = [(rank, weight, tag) for tag, (rank, weight, cid) in appearances.items()
                  if cid != category and rank <= 1]
        others.sort(key=lambda item: (item[0], -item[1], item[2]))
        tags.extend(tag for _rank, _weight, tag in others)
    for token in _content_tokens(title):
        if not _covered(token, tags):
            tags.append(token)
    if len(fit_tags(tags)) < 3:
        for token in _frequent(transcript):
            if not _covered(token, tags):
                tags.append(token)
    return fit_tags(tags)


def classify(title, description='', transcript=''):
    """Return categoryId, English category name, tags, and where the category came from.

    `categorySource` is `inferred` when some topic cleared the bar, and
    `default` when nothing did (Education).
    """
    title, description, transcript = _clip(title), _clip(description), _clip(transcript)
    scores = score_categories(title, description, transcript)
    best = max(CATEGORIES, key=lambda cid: (scores[cid], TIE_BREAK[cid]))
    if scores[best] < MIN_SCORE:
        chosen = DEFAULT_CATEGORY
        source = 'default'
    else:
        chosen = best
        source = 'inferred'
    tags = suggest_tags(title, description, transcript, category=chosen)
    return {
        'categoryId': chosen,
        'category': CATEGORIES[chosen],
        'tags': tags,
        'categorySource': source,
    }


def _chapter_text(metadata):
    lines = []
    for chapter in metadata.get('chapters') or []:
        if isinstance(chapter, dict):
            title = str(chapter.get('title') or '').strip()
            if title:
                lines.append(title)
    return '\n'.join(lines)


def _file_category(metadata):
    if not isinstance(metadata, dict) or 'categoryId' not in metadata:
        return None
    if metadata['categoryId'] is None:
        return None
    text = str(metadata['categoryId']).strip()
    return text or None


def _env_value(environ, name):
    if environ is None:
        environ = os.environ
    return str(environ.get(name, '') or '').strip()


def resolve_category_and_tags(metadata, transcript='', category=None, tags=None, environ=None):
    """Apply overrides, then fill whatever is still empty from the text.

    Category and tags are independent. Forcing a category does not throw away
    tags inferred from the transcript, and the other way around.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    title = str(metadata.get('title') or '')
    description = str(metadata.get('description') or '')
    chapters = _chapter_text(metadata)
    if chapters:
        description = f'{description}\n{chapters}' if description else chapters
    inferred = classify(title, description, transcript or '')

    file_category = _file_category(metadata)
    env_category = _env_value(environ, 'YOUTUBE_CATEGORY_ID')
    if category is not None and str(category).strip():
        chosen_category = category_id(category)
        category_source = 'argument'
    elif file_category:
        chosen_category = category_id(file_category)
        category_source = 'file'
    elif env_category:
        chosen_category = category_id(env_category)
        category_source = 'env'
    else:
        chosen_category = inferred['categoryId']
        category_source = inferred['categorySource']

    env_tags = _env_value(environ, 'YOUTUBE_TAGS')
    if tags is not None:
        chosen_tags = normalize_tags(tags, required=True)
        tags_source = 'argument'
    elif 'tags' in metadata and metadata.get('tags') is not None:
        chosen_tags = normalize_tags(metadata['tags'], required=True)
        tags_source = 'file'
    elif env_tags:
        chosen_tags = normalize_tags(env_tags, required=True)
        tags_source = 'env'
    else:
        chosen_tags = list(inferred['tags'])
        tags_source = 'inferred'

    return {
        'categoryId': chosen_category,
        'category': CATEGORIES[chosen_category],
        'tags': chosen_tags,
        'categorySource': category_source,
        'tagsSource': tags_source,
    }
