# 음성 타이핑 영상 생성기

`files/audio.m4a` (또는 `files` 안의 유일한 음성 파일)를 로컬 Whisper로 인식하고, 음성의 단어별 타임스탬프에 맞춰 한 글자씩 등장하는 MP4를 만듭니다. 기본 7줄이 차면 다음 글자가 등장할 때 화면을 지웁니다. 무음 구간에는 마지막 텍스트를 유지합니다.

## 설치 및 실행 (macOS)

uv와 FFmpeg가 필요합니다 (`brew install uv ffmpeg`). Python 버전은 `.python-version`, 의존성은 `pyproject.toml`과 `uv.lock`으로 관리합니다. `uv sync --locked`가 가상환경을 생성하고 잠긴 버전을 설치합니다.

```sh
uv sync --locked
# 짧은 구간으로 품질 확인
uv run scripter.py --duration 20 -o files/preview.mp4
# 전체 음성
uv run scripter.py
```

첫 실행 시 Hugging Face에서 모델을 `.models/`에 다운로드합니다. 기본 실행의 음성 인식은 로컬에서 처리합니다. `--youtube`는 전사문과 썸네일 생성 지시를 OpenAI API로 전송합니다. CPU에서 약 48분 음성의 인식과 렌더링은 시간이 걸릴 수 있습니다. `--model medium`은 더 큰 모델, `--model tiny`는 빠른 시험용 모델입니다.

음성 인식과 렌더링 단계마다 `렌더링: 30.0 / 120.0초 (25.0%)`처럼 처리한 음성/영상 길이와 진행률을 출력합니다. 구간 지정 시 선택한 구간 길이를 기준으로 계산합니다. 인식은 인식된 구간마다, 렌더링은 영상 10초마다 출력하며 각 단계가 끝나면 100%를 표시합니다. 이 진행률은 처리 분량 기준이며, 실제 소요 시간의 비율은 아닙니다.

## 결과와 조정

### 입력 파일 지정

`-i` 또는 `--input`으로 음성 파일 경로를 지정합니다. 생략하면 `files/audio.m4a` (또는 `files` 안의 유일한 음성 파일)를 사용합니다.

```sh
uv run scripter.py -i /path/to/recording.m4a -o files/video.mp4
uv run scripter.py --input "./녹음 파일.m4a" --duration 02:00 -o files/preview.mp4
# 기존 위치 인자 방식도 지원
uv run scripter.py recording.m4a -o files/video.mp4
```

위치 인자와 `--input`은 함께 지정할 수 없습니다.

### 생성 구간 지정

```sh
# 초반 2분 (120 또는 02:00)
uv run scripter.py --duration 02:00 -o files/first-2min.mp4
# 원본 음성의 5분부터 7분 30초까지
uv run scripter.py --start 05:00 --end 07:30 -o files/section.mp4
# 5분부터 2분 동안
uv run scripter.py --start 05:00 --duration 02:00 -o files/section-2min.mp4
```

시각은 초, `MM:SS`, `HH:MM:SS` 형식을 지원합니다. `--end`와 `--duration`은 함께 사용할 수 없습니다. 종료 시각이 원본 길이를 넘으면 음성 끝까지만 생성하며, 시작만 지정하면 그 시각부터 끝까지 생성합니다. 지정한 구간만 음성 인식하므로 전체 음성을 먼저 인식할 필요가 없습니다.

JSON의 시각은 **원본 음성 기준**입니다. 구간 영상의 JSON을 다시 사용할 때도 동일한 `--start`와 `--end`(또는 `--duration`)를 지정하세요. 전체 음성의 JSON에서 원하는 구간만 재렌더링할 수도 있습니다. 구간 경계에 걸친 단어는 포함하며, 발화 중간에서 자르면 해당 단어가 일부만 들릴 수 있습니다.

```sh
uv run scripter.py --transcript files/section.words.json --start 05:00 --end 07:30 -o files/section-revised.mp4
```

- `files/video.mp4`: 1920×1080, 30fps, H.264 + AAC, 16:9.
- `files/video.txt`: 추출 텍스트.
- `files/video.words.json`: 단어별 시작/종료 시각. 텍스트나 시각을 수정한 뒤 재렌더링할 수 있습니다.
- `--font-size 80`: 글꼴 크기(px), 기본값 80. 더 크게 표시하려면 96 또는 112를 지정하세요.
- `--lines 5`: 화면에 쌓이는 최대 줄 수. 글꼴이 커지면 화면 높이에 맞춰 실제 줄 수를 줄입니다. 줄바꿈도 글꼴의 실제 폭에 맞춰 자동 조정됩니다.
- `--fps 60`: 더 촘촘한 시간 표현. 일반 타이핑 영상은 30fps 권장.
- `--font /path/to/font.ttf`: 다른 운영체제에서는 한글 폰트를 지정하세요.

```sh
uv run scripter.py --transcript files/video.words.json -o files/revised.mp4
```

글꼴을 크게 하여 초반 2분을 생성하려면:

```sh
uv run scripter.py --duration 02:00 --font-size 96 -o files/first-2min-large.mp4
```

이미 추출한 JSON을 재사용하면 음성 인식을 생략하고 글꼴 크기만 바꿀 수 있습니다:

```sh
uv run scripter.py --duration 02:00 --transcript files/first-2min.words.json --font-size 96 -o files/first-2min-large.mp4
```

단어 내부의 글자 시각은 단어 발화 구간을 균등 분할한 추정값입니다. Whisper의 오인식 또는 타이밍 오류는 JSON을 수정하거나 더 큰 모델로 재인식하세요. 화면 폭은 폰트의 실제 글자 폭으로 계산하며, 한글은 완성형 글자 단위로 나타납니다 (자모 입력 애니메이션은 아님).

해상도와 프레임레이트는 `--width`, `--height`, `--fps`로 변경할 수 있습니다. 출력 경로에 파일이 있으면 덮어씁니다.

유튜브 공식 권장 형식: https://support.google.com/youtube/answer/1722171

## 수식 받아쓰기

말로 읽은 수식을 영상에서 조판된 수식으로 보여 줍니다. [mark-vector](https://github.com/iasandcb/mark-vector)의 수식 받아쓰기와 같은 규칙입니다.

- "수식시작"이라고 말하면 수식 구간이 시작되고 "수식끝"에서 끝납니다. 그 사이의 말은 단어가 끝날 때마다 AsciiMath로 바뀌고, 지금까지의 수식이 조판되어 화면에 나타납니다. 수식은 가운데 정렬하며, 화면 폭이나 한 페이지보다 크면 줄여서 그립니다.
- 규칙은 mark-vector의 **공용 수식 말**(`spoken-math.csv`)과 같은 CSV입니다. mark-vector에서 내려받아 `files/spoken-math.csv`에 두면 자동으로 쓰고, 다른 경로는 `--spoken-math 경로`로 지정합니다. 파일이 없으면 이전처럼 글자만 그립니다.
- 예: "넓이는 수식 시작 적분 밑 영 위 일 엑스 승 이 디엑스 수식끝 입니다" → ∫₀¹ x² dx
- `files/<이름>.txt`에는 수식 구간이 `$$ … $$` 문단으로 들어갑니다. `.words.json`은 인식 결과 그대로라서, 규칙을 고친 뒤 `--transcript`로 다시 렌더링할 수 있습니다.

수식 조판에는 Node.js가 필요합니다(AsciiMath → LaTeX는 asciimath-parser, 그림은 MathJax + resvg, `math_render.mjs`).

```sh
brew install node
npm install   # scripter 폴더에서 한 번
uv run scripter.py --spoken-math files/spoken-math.csv
```

규칙에 없는 한글(예: "삼 분의 일"의 "분의")은 수식 글꼴에 한글이 없어 수식 안에서 보이지 않습니다. 자주 쓰는 말은 규칙에 추가하세요.

## 유튜브 업로드 자료 생성

입력 음성을 `files`에 넣고 실행하세요. OpenAI API 키는 프로젝트 루트(`scripter.py` 옆)의 `.env` 또는 환경 변수에 설정하며 API 사용 요금이 발생합니다. 실행 시 `.env`를 자동으로 읽습니다. 기존 환경 변수는 `.env`보다 우선합니다.

`.env.example`을 `.env`로 복사하고 키를 입력하세요. `.env`는 Git에서 제외됩니다.

```dotenv
OPENAI_API_KEY=your-api-key
```

```sh
uv run scripter.py --youtube
# 입력, 출력, 전사 구간 지정도 함께 사용 가능
uv run scripter.py -i files/20261007.m4a --duration 02:00 --youtube
```

기본적으로 제목·설명은 `gpt-4.1-mini`, 썸네일은 `gpt-image-1`로 생성합니다. `--youtube-model`, `--youtube-image-model`로 각각 변경할 수 있습니다. 이미지 모델은 GPT Image API의 `1536x1024` PNG 생성을 지원해야 합니다.

예를 들어 입력이 `files/20261007.m4a`이면 다음 결과가 저장됩니다.

- `files/20261007.mp4`, `.txt`, `.words.json`: 영상과 전사문.
- `files/20261007.youtube.txt`: 복사해서 업로드할 제목과 설명.
- `files/20261007.youtube.json`: 제목, 설명, 썸네일 생성 지시.
- `files/20261007.thumbnail-prompt.txt`: 썸네일 생성 지시.
- `files/20261007.thumbnail.png`: 1280×720 썸네일. 생성 이미지를 중앙 기준 16:9로 잘라 저장합니다.

선택한 구간의 전사문 전체에서 자료를 생성합니다. 썸네일 요청에 실패해도 영상·전사문·제목·설명은 보존합니다. 생성된 내용은 확인 후 사용하세요. 실제 유튜브 업로드와 계정 인증은 아직 포함하지 않습니다.

API 구현 참고: [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Image generation API](https://developers.openai.com/api/reference/resources/images/methods/generate).

## 개발 및 검증

```sh
uv run python -m unittest
```
