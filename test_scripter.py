import unittest
from scripter import timeline, typing_stream, document_text, Math, parse_time, select_range, crop_words
from spoken_math import SpokenMath
import argparse
import base64
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripter import generate_youtube, main


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
                 patch('scripter.probe', return_value=10), patch('scripter.render') as render, \
                 patch('scripter.generate_youtube') as youtube:
                main()
            self.assertEqual(render.call_args.args[1], Path('files/recording.mp4'))
            self.assertEqual(youtube.call_args.args[0], '선택한 내용')


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


SPOKEN = SpokenMath.from_csv("""수식시작, $$
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
        self.assertEqual(SPOKEN.to_asciimath('엑스 승 2 더하기 라지 에프.'), 'x ^ 2 + F')
        self.assertEqual(SPOKEN.to_asciimath('F 는 X 이고 라지 F'), 'f = x\nF')

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


if __name__ == '__main__':
    unittest.main()
