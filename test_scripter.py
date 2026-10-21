import unittest
from scripter import timeline, typing_stream, document_text, Math, SpokenMath, parse_time, select_range, crop_words
import argparse
import base64
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from scripter import (
    generate_youtube, main, fetch_spoken_math, edit_stream, timed_text, chapter_lines,
    load_glossary, glossary_prompt, word_change_ratio, transcript_chunks, correction_diff,
    correct_transcript, commit_correction, apply_spoken_math_markers, transcribe, transcribe_mlx,
    CORRECT_INSTRUCTIONS)


class DotenvTests(unittest.TestCase):
    def test_main_loads_project_dotenv_and_preserves_environment(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '.env').write_text('OPENAI_API_KEY="test-from-file"\nSCRIPTER_TEST_VALUE=file\n')
            with patch('scripter.__file__', str(root / 'scripter.py')), \
                 patch.dict(os.environ, {'SCRIPTER_TEST_VALUE': 'shell'}, clear=True), \
                 patch('sys.argv', ['scripter.py', '--help']), patch('sys.stdout'):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 0)
                self.assertEqual(os.environ['OPENAI_API_KEY'], 'test-from-file')
                self.assertEqual(os.environ['SCRIPTER_TEST_VALUE'], 'shell')


class YoutubeTests(unittest.TestCase):
    def test_materials_and_thumbnail(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (1536, 1024), 'blue').save(buffer, format='PNG')
        metadata = {'title': 'AI와 일상', 'description': 'AI 활용 경험을 소개합니다.',
                    'thumbnail_prompt': '일상 속 AI를 표현하는 그림'}
        responses = [
            {'status': 'completed', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps(metadata)}]}]},
            {'data': [{'b64_json': base64.b64encode(buffer.getvalue()).decode()}]}]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'video.mp4'
            with patch('scripter.openai_request', side_effect=responses) as api:
                generate_youtube('오늘은 AI 활용 경험을 이야기합니다.', output, 'text-model', 'image-model')
            self.assertEqual(api.call_args_list[0].args[1]['input'], '오늘은 AI 활용 경험을 이야기합니다.')
            self.assertEqual(json.loads(output.with_suffix('.youtube.json').read_text()), metadata)
            self.assertIn(metadata['description'], output.with_suffix('.youtube.txt').read_text())
            with Image.open(output.with_suffix('.thumbnail.png')) as image:
                self.assertEqual(image.size, (1280, 720))

    def test_description_ends_with_valid_chapters(self):
        metadata = {'title': '제목', 'description': '설명', 'thumbnail_prompt': '그림', 'chapters': [
            {'start': '00:05', 'title': '인사'}, {'start': '01:30', 'title': '바둑'},
            {'start': '01:35', 'title': '너무 가까움'}, {'start': '03:00', 'title': '미적분'}]}
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(metadata)}]}]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'video.mp4'
            with patch('scripter.openai_request', side_effect=[response, RuntimeError('이미지 생략')]) as api:
                with self.assertRaises(RuntimeError):
                    generate_youtube('전사문', output, 'text-model', 'image-model', timed='[00:00] 전사문', duration=240)
            request = api.call_args_list[0].args[1]
            self.assertEqual(request['input'], '[00:00] 전사문')
            self.assertIn('chapters', request['text']['format']['schema']['required'])
            self.assertEqual(output.with_suffix('.youtube.txt').read_text(),
                             '제목\n\n설명\n\n00:00 인사\n01:30 바둑\n03:00 미적분\n')
            self.assertEqual(json.loads(output.with_suffix('.youtube.json').read_text())['chapters'][0],
                             {'start': '00:00', 'title': '인사'})

    def test_image_failure_preserves_metadata(self):
        metadata = {'title': '제목', 'description': '설명', 'thumbnail_prompt': '그림'}
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(metadata)}]}]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'video.mp4'
            with patch('scripter.openai_request', side_effect=[response, RuntimeError('API 실패')]):
                with self.assertRaises(RuntimeError):
                    generate_youtube('전사문', output, 'text-model', 'image-model')
            self.assertTrue(output.with_suffix('.youtube.txt').is_file())
            self.assertFalse(output.with_suffix('.thumbnail.png').exists())

    def test_cli_thumbnail_uses_edited_prompt_only(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (1536, 1024), 'red').save(buffer, format='PNG')
        response = {'data': [{'b64_json': base64.b64encode(buffer.getvalue()).decode()}]}
        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'talk.m4a', Path(folder) / 'talk.mp4'
            output.with_suffix('.thumbnail-prompt.txt').write_text('고친 그림 지시\n')
            output.with_suffix('.youtube.json').write_text(json.dumps(
                {'title': '제목', 'description': '설명', 'thumbnail_prompt': '예전 지시'}))
            with patch('sys.argv', ['scripter.py', '-i', str(audio), '-o', str(output), '--thumbnail']), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), \
                 patch('scripter.openai_request', return_value=response) as api, \
                 patch('scripter.render') as render, patch('scripter.transcribe') as transcribe:
                main()
            self.assertEqual(api.call_count, 1)
            self.assertEqual(api.call_args.args[1]['prompt'], '고친 그림 지시')
            render.assert_not_called()
            transcribe.assert_not_called()
            self.assertEqual(json.loads(output.with_suffix('.youtube.json').read_text())['thumbnail_prompt'], '고친 그림 지시')
            with Image.open(output.with_suffix('.thumbnail.png')) as image:
                self.assertEqual(image.size, (1280, 720))

    def test_cli_uses_selected_transcript_and_files_output(self):
        words = [{'text': '제외', 'start': 0, 'end': 1},
                 {'text': '선택한 내용', 'start': 2, 'end': 4}]
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / 'recording.m4a'
            transcript = Path(folder) / 'words.json'
            transcript.write_text(json.dumps(words))
            with patch('sys.argv', ['scripter.py', '-i', str(audio), '--transcript',
                       str(transcript), '--start', '2', '--end', '4', '--youtube']), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), \
                 patch('scripter.shutil.which', return_value='tool'), \
                 patch('scripter.Path.is_file', lambda path: path.name != 'spoken-math.csv'), \
                 patch('scripter.Path.mkdir'), patch('scripter.Path.write_text'), \
                 patch('scripter.fetch_spoken_math', return_value=None), \
                 patch('scripter.probe', return_value=10), patch('scripter.render') as render, \
                 patch('scripter.generate_youtube') as youtube:
                main()
            self.assertEqual(render.call_args.args[1], Path('files/recording.mp4'))
            self.assertEqual(youtube.call_args.args[0], '선택한 내용')


class StageTests(unittest.TestCase):
    def run_main(self, argv, folder, total=10, mlx=False):
        with patch('sys.argv', ['scripter.py', *argv]), \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), \
             patch('scripter.shutil.which', return_value='tool'), \
             patch('scripter.fetch_spoken_math', return_value=None), \
             patch('scripter.Path.is_file', lambda path: path.suffix in ('.m4a', '.ttc') or path.exists()), \
             patch('scripter.probe', return_value=total), patch('scripter.mlx_available', return_value=mlx), \
             patch('scripter.transcribe', return_value=[{'text': '안녕하세요', 'start': 1, 'end': 2}]), \
             patch('scripter.render') as render, patch('scripter.generate_youtube') as youtube:
            main()
        return render, youtube

    def test_no_video_writes_transcript_and_resume_renders_it(self):
        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'audio.m4a', Path(folder) / 'video.mp4'
            render, youtube = self.run_main(['-i', str(audio), '-o', str(output), '--no-video'], folder)
            render.assert_not_called()
            youtube.assert_not_called()
            self.assertEqual(output.with_suffix('.txt').read_text(), '안녕하세요')
            output.with_suffix('.txt').write_text('수정한 내용\n')
            with patch('scripter.transcribe') as transcribe:
                render, youtube = self.run_main(['-i', str(audio), '-o', str(output), '--resume', '--youtube'], folder)
                transcribe.assert_not_called()
            self.assertEqual(render.call_args.args[1], output)
            self.assertEqual(youtube.call_args.args[0], '수정한 내용\n')
            self.assertEqual([item[1] for item in render.call_args.args[2]], list('수정한 내용'))
            self.assertEqual(output.with_suffix('.txt').read_text(), '수정한 내용\n')

    def test_chapters_only_updates_youtube_text(self):
        reply = {'chapters': [{'start': '00:00', 'title': '시작'}, {'start': '00:20', 'title': '중간'},
                              {'start': '00:40', 'title': '끝'}]}
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps(reply)}]}]}
        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'audio.m4a', Path(folder) / 'video.mp4'
            output.with_suffix('.words.json').write_text(json.dumps(
                [{'text': ' 처음.', 'start': 1, 'end': 2}, {'text': ' 다음.', 'start': 30, 'end': 31}]))
            output.with_suffix('.txt').write_text(' 처음.\n고친 다음.\n')
            output.with_suffix('.youtube.json').write_text(json.dumps(
                {'title': '제목', 'description': '설명', 'thumbnail_prompt': '그림', 'chapters': []}))
            with patch('scripter.openai_request', return_value=response) as api:
                render, youtube = self.run_main(['-i', str(audio), '-o', str(output), '--chapters'], folder, total=60)
            render.assert_not_called()
            youtube.assert_not_called()
            self.assertEqual(api.call_args.args[1]['input'], '[00:01] 처음.\n[00:30] 고친 다음.')
            self.assertEqual(output.with_suffix('.youtube.txt').read_text(),
                             '제목\n\n설명\n\n00:00 시작\n00:20 중간\n00:40 끝\n')
            self.assertEqual(json.loads(output.with_suffix('.youtube.json').read_text())['thumbnail_prompt'], '그림')

    def test_engine_choice(self):
        words = [{'text': '안녕하세요', 'start': 1, 'end': 2}]
        with tempfile.TemporaryDirectory() as folder:
            audio = str(Path(folder) / 'audio.m4a')
            for argv, mlx, chosen in [([], True, 'mlx large-v3'),
                                      (['--model', 'large-v3-turbo'], True, 'mlx large-v3-turbo'),
                                      ([], False, 'faster-whisper small'),
                                      (['--engine', 'faster-whisper'], True, 'faster-whisper small')]:
                with patch('scripter.transcribe_mlx', return_value=words) as transcribe_mlx, \
                     patch('sys.stdout', new_callable=io.StringIO) as printed:
                    self.run_main(['-i', audio, '-o', str(Path(folder) / 'video.mp4'), '--no-video', *argv], folder, mlx=mlx)
                self.assertIn(f'음성 인식: {chosen}\n', printed.getvalue())
                self.assertEqual(transcribe_mlx.called, chosen.startswith('mlx'))

    def test_resume_requires_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                self.run_main(['-i', str(Path(folder) / 'audio.m4a'), '-o', str(Path(folder) / 'none.mp4'), '--resume'], folder)

    def test_no_video_rejects_youtube(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('sys.stderr'), self.assertRaises(SystemExit):
                self.run_main(['-i', str(Path(folder) / 'audio.m4a'), '--no-video', '--youtube'], folder)


class ChapterTests(unittest.TestCase):
    def test_timed_text(self):
        stream = typing_stream([{'text': ' 안녕하세요.', 'start': 1, 'end': 2}, {'text': ' 바둑', 'start': 75, 'end': 76},
                                {'text': ' 얘기', 'start': 77, 'end': 78}])
        stream[7:7] = [('open', 3), ('math', 'x', 4), ('math', 'x^2', 5), ('close', 6)]
        self.assertEqual(timed_text(stream), '[00:01] 안녕하세요.\n[00:03] $$ x^2 $$\n[01:15] 바둑 얘기')

    def test_chapter_rules(self):
        chapters = [{'start': start, 'title': title} for start, title in
                    [('00:12', '처음'), ('00:30', '둘'), ('00:35', '가까움'), ('9:99', '잘못'), ('01:00', '셋'), ('01:55', '끝 직전')]]
        self.assertEqual(chapter_lines(chapters, 120), ['00:00 처음', '00:30 둘', '01:00 셋'])
        self.assertEqual(chapter_lines(chapters[:2], 120), [])
        self.assertEqual(chapter_lines([{'start': '0', 'title': 'a'}, {'start': '30:00', 'title': 'b'},
                                        {'start': '1:05:00', 'title': 'c'}], 4000),
                         ['0:00:00 a', '0:30:00 b', '1:05:00 c'])


class RangeTests(unittest.TestCase):
    def test_time_formats(self):
        self.assertEqual(parse_time('120'), 120)
        self.assertEqual(parse_time('02:00'), 120)
        self.assertEqual(parse_time('01:02:03.5'), 3723.5)
        for invalid in ('-1', 'nan', 'inf', '1:60', 'abc', '1:2:3:4'):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_time(invalid)

    def test_range(self):
        self.assertEqual(select_range(1000, 300, duration=120), (300, 420))
        self.assertEqual(select_range(1000, 300, end=450), (300, 450))
        self.assertEqual(select_range(1000, 900, duration=120), (900, 1000))
        for kwargs in ({'start': 1000}, {'start': 100, 'end': 50},
                       {'start': 0, 'duration': 0}, {'start': 0, 'end': 5, 'duration': 2}):
            with self.assertRaises(ValueError):
                select_range(1000, **kwargs)

    def test_crop_and_rebase(self):
        words = [{'text': '가', 'start': 4, 'end': 6},
                 {'text': '나', 'start': 7, 'end': 11},
                 {'text': '다', 'start': 10, 'end': 12}]
        self.assertEqual(crop_words(words, 5, 10), [
            {'text': '가', 'start': 0, 'end': 1},
            {'text': '나', 'start': 2, 'end': 5}])
        self.assertEqual(words[0]['start'], 4)

class FixedFont:
    def getlength(self, text):
        return len(text) * 10

class FetchSpokenMathTests(unittest.TestCase):
    def test_saves_fetched_rules(self):
        body = io.BytesIO(json.dumps({'content': '엑스, x\n'}).encode())
        with tempfile.TemporaryDirectory() as folder, patch('scripter.urlopen', return_value=body), patch('sys.stdout'):
            cache = Path(folder) / 'files' / 'spoken-math.csv'
            self.assertEqual(fetch_spoken_math('https://example.invalid', cache), cache)
            self.assertEqual(cache.read_text(encoding='utf-8'), '엑스, x\n')

    def test_falls_back_to_saved_rules_offline(self):
        from urllib.error import URLError
        with tempfile.TemporaryDirectory() as folder, patch('scripter.urlopen', side_effect=URLError('offline')), \
             patch('sys.stdout'):
            cache = Path(folder) / 'spoken-math.csv'
            self.assertIsNone(fetch_spoken_math('https://example.invalid', cache))
            cache.write_text('엑스, x\n', encoding='utf-8')
            self.assertEqual(fetch_spoken_math('https://example.invalid', cache), cache)
            self.assertEqual(cache.read_text(encoding='utf-8'), '엑스, x\n')


class TimelineTests(unittest.TestCase):
    def test_character_timing(self):
        events = timeline(typing_stream([{'text': '가나다', 'start': 2, 'end': 5}]), FixedFont(), 100, 2)
        self.assertEqual(events, [(2, ('가',)), (3, ('가나',)), (4, ('가나다',))])

    def test_clear_before_overflow(self):
        events = timeline(typing_stream([{'text': '가나다라마', 'start': 0, 'end': 5}]), FixedFont(), 20, 2)
        self.assertEqual(events[3][1], ('가나', '다라'))
        self.assertEqual(events[4], (4, ('마',)))

    def test_silence_and_overlapping_timestamps(self):
        events = timeline(typing_stream([
            {'text': '가', 'start': 1, 'end': 2},
            {'text': '나', 'start': 5, 'end': 6},
            {'text': '다', 'start': 5.5, 'end': 7},
        ]), FixedFont(), 100, 2)
        self.assertEqual([at for at, _ in events], [1, 5, 6])


SPOKEN = SpokenMath("""수식시작, $$
수식끝, $$
엑스, x
승, ^
더하기, +
라지에프, F
에프, f
이고, \\n
는, =
""")


class SpokenMathTests(unittest.TestCase):
    def test_conversion(self):
        # Converted by asciimath-markdown, the converter mark-vector uses.
        self.assertEqual(SPOKEN.convert(['엑스 승 2 더하기 라지 에프.', 'F 는 X 이고 라지 F']),
                         ['x ^ 2 + F', 'f = x\nF'])

    def test_block_commands_across_words(self):
        # Whisper splits "수식 시작" into two words; the command still spans them.
        words = [{'text': ' 정리하면', 'start': 0, 'end': 1},
                 {'text': ' 수식', 'start': 1, 'end': 2}, {'text': ' 시작', 'start': 2, 'end': 3},
                 {'text': ' 엑스', 'start': 3, 'end': 4}, {'text': ' 승', 'start': 4, 'end': 5},
                 {'text': ' 2', 'start': 5, 'end': 6},
                 {'text': ' 수식끝', 'start': 6, 'end': 7}, {'text': ' 입니다', 'start': 7, 'end': 8}]
        stream = typing_stream(words, SPOKEN)
        self.assertEqual([item[1] for item in stream if item[0] == 'math'], ['x', 'x ^', 'x ^ 2'])
        self.assertEqual(document_text(stream), ' 정리하면\n\n$$\nx ^ 2\n$$\n\n입니다')
        pages = [page for _, page in timeline(stream, FixedFont(), 100, 5)]
        self.assertEqual(pages[-1], ('정리하면 ', Math('x ^ 2'), '입니다'))

    def test_tall_formula_moves_to_a_new_page(self):
        words = [{'text': '가나', 'start': 0, 'end': 1}, {'text': ' 수식시작', 'start': 1, 'end': 2},
                 {'text': ' 엑스', 'start': 2, 'end': 3}, {'text': ' 수식끝', 'start': 3, 'end': 4}]
        pages = [page for _, page in timeline(typing_stream(words, SPOKEN), FixedFont(), 100, 2,
                                              math_lines=lambda source: 2)]
        self.assertEqual(pages[-1], (Math('x'),))

    def test_no_vocabulary_is_plain_text(self):
        stream = typing_stream([{'text': '수식시작 엑스', 'start': 0, 'end': 1}])
        self.assertEqual(document_text(stream), '수식시작 엑스')


class EditStreamTests(unittest.TestCase):
    def stream(self, *words):
        return typing_stream([{'text': text, 'start': start, 'end': end} for text, start, end in words])

    def test_unchanged_text_keeps_stream(self):
        stream = self.stream((' 안녕', 0, 1), (' 하세요', 2, 3))
        self.assertIs(edit_stream(stream, document_text(stream)), stream)

    def test_changed_word_takes_replaced_time(self):
        stream = self.stream((' 두도', 0, 1), (' 없이', 2, 3))
        edited = edit_stream(stream, ' 두서 없이')
        self.assertEqual(document_text(edited), ' 두서 없이')
        self.assertEqual([at for kind, char, at in edited if char == '없'], [2 + 1 / 3])
        self.assertTrue(all(0 <= item[-1] <= 1 for item in edited[:3]))

    def test_line_per_sentence_reads_as_spaces(self):
        stream = self.stream((' 안녕하세요.', 0, 1), (' 정말?', 2, 3), (' 3.5', 4, 5), (' 끝', 6, 7))
        self.assertEqual(document_text(stream), ' 안녕하세요.\n정말?\n3.5 끝')
        edited = edit_stream(stream, ' 안녕하세요.\n정말!\n3.5 끝')
        self.assertNotIn('\n', [item[1] for item in edited])
        self.assertEqual(''.join(item[1] for item in edited), ' 안녕하세요. 정말! 3.5 끝')
        self.assertEqual(edited[-1], stream[-1])

    def test_corrected_formula(self):
        stream = [('text', '넓', 0), ('text', '이', 0.5), ('open', 1), ('math', '라지 fx', 2),
                  ('close', 3), ('text', ' ', 3), ('text', '끝', 4)]
        edited = edit_stream(stream, '넓이\n\n$$\nF(x) = f(x)\n$$\n\n끝')
        self.assertEqual([item for item in edited if item[0] != 'text'],
                         [('open', 1), ('math', 'F(x)', 1 + 2 / 3), ('math', 'F(x) =', 1 + 4 / 3),
                          ('math', 'F(x) = f(x)', 3), ('close', 3)])
        self.assertEqual(edited[-1], ('text', '끝', 4))

    def test_text_turned_into_formula(self):
        stream = self.stream((' 값은', 0, 1), (' 엑스', 2, 3), (' 제곱', 3, 4), (' 이다', 5, 6))
        edited = edit_stream(stream, ' 값은\n\n$$\nx^2\n$$\n\n이다')
        marks = [item for item in edited if item[0] != 'text']
        self.assertEqual([item[:-1] for item in marks], [('open',), ('math', 'x^2'), ('close',)])
        for item, at in zip(marks, (2, 3 + 2 / 3, 3 + 2 / 3)):
            self.assertAlmostEqual(item[-1], at)
        self.assertEqual(document_text(edited), ' 값은\n\n$$\nx^2\n$$\n\n이다')


def completed(text):
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps({'text': text}, ensure_ascii=False)}]}]}


def transcript_of(payload):
    return payload['input'].split('\n\n전사문:\n', 1)[1]


class GlossaryTests(unittest.TestCase):
    def test_load_and_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'glossary.txt'
            path.write_text('# 주석\n\n 마크벡터 \n\n수식시작\n', encoding='utf-8')
            self.assertEqual(load_glossary(path), ['마크벡터', '수식시작'])
        prompt, truncated = glossary_prompt(['가나다', '라마바', '사아자'], limit=8)
        self.assertEqual(prompt, '가나다, 라마바')
        self.assertTrue(truncated)
        prompt, truncated = glossary_prompt(['가', '나'], limit=10)
        self.assertEqual(prompt, '가, 나')
        self.assertFalse(truncated)
        self.assertEqual(glossary_prompt([], limit=10), (None, False))
        prompt, truncated = glossary_prompt(['가나다라마바사'], limit=4)
        self.assertEqual(prompt, '가나다라')
        self.assertTrue(truncated)

    def test_mlx_prompt_keeps_silence_guards(self):
        # mlx applies initial_prompt to the first window only while
        # condition_on_previous_text stays False, so the hint is not fed
        # back and cannot restart the silence-repetition loop.
        mlx, hub = MagicMock(), MagicMock()
        mlx.transcribe.return_value = {'segments': []}
        hub.snapshot_download.return_value = '/model'
        with patch.dict(sys.modules, {'mlx_whisper': mlx, 'huggingface_hub': hub}), patch('sys.stdout'):
            self.assertEqual(transcribe_mlx(Path('audio.wav'), 'large-v3', 'ko', Path('.models'), 3, prompt='마크벡터'), [])
        kwargs = mlx.transcribe.call_args.kwargs
        self.assertFalse(kwargs['condition_on_previous_text'])
        self.assertEqual(kwargs['hallucination_silence_threshold'], 2.0)
        self.assertTrue(kwargs['word_timestamps'])
        self.assertFalse(kwargs['verbose'])
        self.assertEqual(kwargs['initial_prompt'], '마크벡터')
        mlx.transcribe.reset_mock()
        with patch.dict(sys.modules, {'mlx_whisper': mlx, 'huggingface_hub': hub}), patch('sys.stdout'):
            transcribe_mlx(Path('audio.wav'), 'large-v3', 'ko', Path('.models'), 3)
        self.assertNotIn('initial_prompt', mlx.transcribe.call_args.kwargs)
        self.assertFalse(mlx.transcribe.call_args.kwargs['condition_on_previous_text'])

    def test_faster_whisper_uses_hotwords_for_every_window(self):
        model = MagicMock()
        model.transcribe.return_value = ((), None)
        fw = MagicMock()
        fw.WhisperModel.return_value = model
        base = dict(language='ko', word_timestamps=True, vad_filter=True, beam_size=5)
        with patch.dict(sys.modules, {'faster_whisper': fw}), patch('sys.stdout'):
            transcribe(Path('audio.wav'), 'small', 'ko', Path('.models'), 3)
        self.assertEqual(model.transcribe.call_args.kwargs, base)
        self.assertEqual(fw.WhisperModel.call_args.kwargs['device'], 'cpu')
        model.transcribe.reset_mock()
        with patch.dict(sys.modules, {'faster_whisper': fw}), patch('sys.stdout'):
            transcribe(Path('audio.wav'), 'small', 'ko', Path('.models'), 3, prompt='마크벡터')
        self.assertEqual(model.transcribe.call_args.kwargs, dict(base, hotwords='마크벡터'))
        self.assertNotIn('initial_prompt', model.transcribe.call_args.kwargs)
        self.assertNotIn('condition_on_previous_text', model.transcribe.call_args.kwargs)

    def test_modal_hotwords_match_local_faster_whisper(self):
        from modal_whisper import recognition_options
        self.assertEqual(recognition_options('ko'),
                         dict(language='ko', word_timestamps=True, vad_filter=True, beam_size=5))
        self.assertEqual(recognition_options('ko', '마크벡터')['hotwords'], '마크벡터')
        self.assertNotIn('initial_prompt', recognition_options('ko', '마크벡터'))


class CorrectionTests(unittest.TestCase):
    def test_word_ratio_and_diff(self):
        self.assertEqual(word_change_ratio('가 나 다', '가 나 다'), 0)
        self.assertEqual(word_change_ratio('  가  나', '가 나'), 0)
        self.assertAlmostEqual(word_change_ratio('가 나 다 라 마', '가 너 다 라 마'), 0.2)
        self.assertEqual(correction_diff('가\n나\n다', '가\n너\n다'), '-나\n+너')
        self.assertEqual(correction_diff('같은 줄', '같은 줄'), '')

    def test_chunks_keep_math_blocks_whole(self):
        text = '앞\n\n$$\nx ^ 2\n$$\n\n' + ('뒤입니다. ' * 30)
        chunks = transcript_chunks(text, limit=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), text)
        for chunk in chunks:
            self.assertEqual(chunk.count('$$') % 2, 0)

    def test_corrects_in_order_and_passes_glossary(self):
        text = ''.join(f'문장 {index} 입니다.\n' for index in range(8))
        seen = []

        def fake(endpoint, payload):
            self.assertEqual(endpoint, 'responses')
            self.assertEqual(payload['model'], 'gpt-6-luna')
            self.assertIn('순서를 바꾸', payload['instructions'])
            self.assertIn('요약', payload['instructions'])
            self.assertIn('띄어쓰기', payload['instructions'])
            self.assertIn('수식시작', payload['instructions'])
            self.assertIn('수식끝', payload['instructions'])
            self.assertIn('마크벡터', payload['input'])
            self.assertEqual(payload['instructions'], CORRECT_INSTRUCTIONS)
            chunk = transcript_of(payload)
            seen.append(chunk)
            return completed(chunk.replace('입니다', '이다'))

        with patch('scripter.CORRECT_CHUNK_CHARS', 40), patch('scripter.openai_request', side_effect=fake), \
             patch('sys.stdout'):
            corrected, ratio = correct_transcript(text, ['마크벡터'], 'gpt-6-luna')
        self.assertGreater(len(seen), 1)
        self.assertEqual(''.join(seen), text)
        self.assertNotIn('입니다', corrected)
        self.assertIn('이다', corrected)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 0.5)

    def test_rejects_a_rewritten_or_broken_chunk_and_retries(self):
        original = '앞 말\n\n$$\nx\n$$\n\n뒤 문장'
        calls = {'n': 0}

        def rewrite(endpoint, payload):
            calls['n'] += 1
            return completed('전혀 다른 긴 문장으로 다시 썼습니다 그리고 더')

        with patch('scripter.openai_request', side_effect=rewrite), patch('sys.stdout'):
            corrected, ratio = correct_transcript(original, [], 'gpt-6-luna')
        self.assertEqual(corrected, original)
        self.assertEqual(ratio, 0)
        self.assertEqual(calls['n'], 1)

        def drop_math(endpoint, payload):
            return completed('앞 말 뒤 문장')

        with patch('scripter.openai_request', side_effect=drop_math), patch('sys.stdout'):
            corrected, ratio = correct_transcript(original, [], 'gpt-6-luna')
        self.assertEqual(corrected, original)
        self.assertEqual(ratio, 0)

        def tweak_formula(endpoint, payload):
            return completed(transcript_of(payload).replace('말', '것').replace('$$\nx\n$$', '$$\ny\n$$'))

        with patch('scripter.openai_request', side_effect=tweak_formula), patch('sys.stdout'):
            corrected, ratio = correct_transcript(original, [], 'gpt-6-luna')
        self.assertIn('$$\nx\n$$', corrected)
        self.assertIn('것', corrected)
        self.assertNotIn('$$\ny\n$$', corrected)

        def fail_then_ok(endpoint, payload):
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('OpenAI responses 연결 실패. 네트워크를 확인하세요.')
            return completed(transcript_of(payload).replace('말', '것'))

        calls['n'] = 0
        with patch('scripter.openai_request', side_effect=fail_then_ok), patch('sys.stdout'):
            corrected, ratio = correct_transcript(original, [], 'gpt-6-luna')
        self.assertEqual(calls['n'], 2)
        self.assertIn('것', corrected)

        with patch('scripter.openai_request', side_effect=RuntimeError('OpenAI responses 요청 실패 (HTTP 401).')), \
             patch('sys.stdout'):
            with self.assertRaises(RuntimeError):
                correct_transcript(original, [], 'gpt-6-luna')

    def test_threshold_keeps_or_forces(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, document = Path(folder) / 'v.raw.txt', Path(folder) / 'v.txt'
            with patch('sys.stdout', new_callable=io.StringIO) as printed:
                kept = commit_correction('원문', '교정', 0.2, 0.1, False, raw, document)
            self.assertEqual(kept, '원문')
            self.assertEqual(document.read_text(encoding='utf-8'), '원문')
            self.assertEqual(raw.read_text(encoding='utf-8'), '원문')
            self.assertIn('원본 전사문을 유지합니다', printed.getvalue())
            self.assertIn('-원문', printed.getvalue())
            with patch('sys.stdout'):
                kept = commit_correction('원문', '교정', 0.2, 0.1, True, raw, document)
            self.assertEqual(kept, '교정')
            self.assertEqual(document.read_text(encoding='utf-8'), '교정')
            with patch('sys.stdout', new_callable=io.StringIO) as printed:
                kept = commit_correction('원문', '원문', 0, 0.1, False, raw, document)
            self.assertEqual(kept, '원문')
            self.assertIn('바꿀 줄이 없습니다', printed.getvalue())

    def test_markers_become_formulas_and_still_align(self):
        plain = '값은 수식시작 엑스 수식끝 입니다'
        converted = apply_spoken_math_markers(plain, SPOKEN)
        self.assertNotIn('수식시작', converted)
        self.assertIn('$$', converted)
        self.assertIn('x', converted)
        self.assertIn('값은', converted)
        self.assertIn('입니다', converted)
        kept = '앞\n\n$$\nF(x)\n$$\n\n뒤'
        self.assertEqual(apply_spoken_math_markers(kept, SPOKEN), kept)
        self.assertEqual(apply_spoken_math_markers(plain, None), plain)
        words = [{'text': '값은', 'start': 0, 'end': 1}, {'text': ' 수직시작', 'start': 1, 'end': 2},
                 {'text': ' 엑스', 'start': 2, 'end': 3}, {'text': ' 수직끝', 'start': 3, 'end': 4},
                 {'text': ' 입니다', 'start': 4, 'end': 5}]
        stream = typing_stream(words)
        original = document_text(stream)
        corrected = apply_spoken_math_markers(original.replace('수직시작', '수식시작').replace('수직끝', '수식끝'), SPOKEN)
        edited = edit_stream(stream, corrected)
        maths = [item for item in edited if item[0] == 'math']
        self.assertTrue(maths)
        self.assertTrue(all(1 <= item[-1] <= 4 for item in maths))
        self.assertIn('입니다', ''.join(item[1] for item in edited if item[0] == 'text'))


class CorrectionCliTests(unittest.TestCase):
    def stage(self, argv, words=None):
        transcribe = MagicMock(return_value=words or [{'text': '안녕하세요', 'start': 1, 'end': 2}])
        stdout = io.StringIO()
        with patch('sys.argv', ['scripter.py', *argv]), \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), \
             patch('scripter.shutil.which', return_value='tool'), \
             patch('scripter.fetch_spoken_math', return_value=None), \
             patch('scripter.Path.is_file', lambda path: path.suffix in ('.m4a', '.ttc') or path.exists()), \
             patch('scripter.probe', return_value=30), patch('scripter.mlx_available', return_value=False), \
             patch('scripter.transcribe', transcribe), patch('scripter.render') as render, \
             patch('scripter.generate_youtube') as youtube, patch('sys.stdout', stdout), patch('sys.stderr'):
            main()
        return transcribe, render, youtube, stdout

    def test_no_video_corrects_and_resume_corrects_existing_text(self):
        sentence = ' '.join('가나다라마바사아자차카타파하거너더러머버서')
        words = [{'text': sentence.split()[0] if index == 0 else ' ' + word, 'start': float(index), 'end': float(index + 1)}
                 for index, word in enumerate(sentence.split())]

        def fix(expected):
            def fake(endpoint, payload):
                self.assertEqual(payload['model'], expected)
                return completed(transcript_of(payload).replace('나', '너', 1))
            return fake

        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'audio.m4a', Path(folder) / 'video.mp4'
            with patch('scripter.openai_request', side_effect=fix('gpt-6-luna')):
                transcribe, render, youtube, stdout = self.stage(
                    ['-i', str(audio), '-o', str(output), '--no-video', '--correct'], words)
            render.assert_not_called()
            youtube.assert_not_called()
            transcribe.assert_called_once()
            self.assertIsNone(transcribe.call_args.kwargs['prompt'])
            original = output.with_suffix('.raw.txt').read_text(encoding='utf-8')
            corrected = output.with_suffix('.txt').read_text(encoding='utf-8')
            self.assertIn('나', original)
            self.assertIn('너', corrected)
            self.assertIn('-', stdout.getvalue())
            self.assertIn('+', stdout.getvalue())
            output.with_suffix('.txt').write_text(original, encoding='utf-8')
            with patch('scripter.openai_request', side_effect=fix('edit-model')):
                transcribe, render, youtube, stdout = self.stage(
                    ['-i', str(audio), '-o', str(output), '--resume', '--correct', '--youtube', '--correct-model', 'edit-model'], words)
            transcribe.assert_not_called()
            self.assertEqual(render.call_args.args[1], output)
            self.assertIn('너', youtube.call_args.args[0])
            self.assertIn('너', ''.join(item[1] for item in render.call_args.args[2] if item[0] == 'text'))
            output.with_suffix('.txt').write_text(original, encoding='utf-8')
            with patch('scripter.openai_request', side_effect=fix('gpt-6-luna')):
                transcribe, render, youtube, _ = self.stage(
                    ['-i', str(audio), '-o', str(output), '--transcript', str(output.with_suffix('.words.json')),
                     '--no-video', '--correct'], words)
            transcribe.assert_not_called()
            render.assert_not_called()
            youtube.assert_not_called()
            self.assertIn('너', output.with_suffix('.txt').read_text(encoding='utf-8'))

    def test_threshold_and_glossary_flags(self):
        sentence = ' '.join(list('가나다라마바사아'))
        words = [{'text': sentence.split()[0] if index == 0 else ' ' + word, 'start': float(index), 'end': float(index + 1)}
                 for index, word in enumerate(sentence.split())]

        def rewrite(endpoint, payload):
            # Two words of eight: over the 10% guard, under the per-chunk reject.
            text = transcript_of(payload)
            return completed(text.replace('나', '너', 1).replace('다', '더', 1))

        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'audio.m4a', Path(folder) / 'video.mp4'
            glossary = Path(folder) / 'terms.txt'
            glossary.write_text('# c\n\n마크벡터\n수식시작\n', encoding='utf-8')
            with patch('scripter.openai_request', side_effect=rewrite):
                transcribe, render, _, stdout = self.stage(
                    ['-i', str(audio), '-o', str(output), '--no-video', '--correct', '--glossary', str(glossary)], words)
            self.assertEqual(transcribe.call_args.kwargs['prompt'], '마크벡터, 수식시작')
            self.assertEqual(output.with_suffix('.txt').read_text(encoding='utf-8'),
                             output.with_suffix('.raw.txt').read_text(encoding='utf-8'))
            self.assertIn('원본 전사문을 유지합니다', stdout.getvalue())
            with patch('scripter.openai_request', side_effect=rewrite):
                _, _, _, stdout = self.stage(
                    ['-i', str(audio), '-o', str(output), '--no-video', '--correct', '--correct-force'], words)
            self.assertIn('--correct-force', stdout.getvalue())
            self.assertIn('너', output.with_suffix('.txt').read_text(encoding='utf-8'))

        with tempfile.TemporaryDirectory() as folder:
            audio, output = Path(folder) / 'audio.m4a', Path(folder) / 'video.mp4'
            glossary = Path(folder) / 'terms.txt'
            glossary.write_text('마크벡터\n', encoding='utf-8')
            def capture_run(argv):
                mock = MagicMock(return_value=[{'text': '안녕하세요', 'start': 1, 'end': 2}])
                with patch('sys.argv', ['scripter.py', *argv]), \
                     patch.dict('os.environ', {'OPENAI_API_KEY': 'test'}), \
                     patch('scripter.shutil.which', return_value='tool'), \
                     patch('scripter.fetch_spoken_math', return_value=None), \
                     patch('scripter.Path.is_file', lambda path: path.suffix in ('.m4a', '.ttc') or path.exists()), \
                     patch('scripter.probe', return_value=10), patch('scripter.mlx_available', return_value=True), \
                     patch('scripter.transcribe_mlx', mock), patch('scripter.render'), \
                     patch('scripter.GLOSSARY_PATH', glossary), patch('sys.stdout'), patch('sys.stderr'):
                    main()
                return mock

            disabled = capture_run(['-i', str(audio), '-o', str(output), '--no-video', '--engine', 'mlx', '--glossary', ''])
            self.assertIsNone(disabled.call_args.kwargs['prompt'])
            used = capture_run(['-i', str(audio), '-o', str(output), '--no-video', '--engine', 'mlx'])
            self.assertEqual(used.call_args.kwargs['prompt'], '마크벡터')

    def test_flag_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = str(Path(folder) / 'audio.m4a')
            for argv in (['--correct', '--thumbnail'], ['--correct-force'],
                         ['--no-video', '--correct', '--glossary', str(Path(folder) / 'missing.txt')]):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    self.stage(['-i', audio, *argv])
            with patch('sys.argv', ['scripter.py', '-i', audio, '--no-video', '--correct']), \
                 patch.dict('os.environ', {'OPENAI_API_KEY': ''}, clear=False), \
                 patch('scripter.shutil.which', return_value='tool'), \
                 patch('scripter.fetch_spoken_math', return_value=None), \
                 patch('scripter.Path.is_file', lambda path: True), \
                 patch('sys.stderr'), self.assertRaises(SystemExit):
                main()


if __name__ == '__main__':
    unittest.main()
