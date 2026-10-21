"""faster-whisper on a Modal GPU, called by scripter.py's --engine modal."""
import modal

app = modal.App('scripter-whisper')
models = modal.Volume.from_name('scripter-whisper-models', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04', add_python='3.11')
         .pip_install('faster-whisper>=1.2,<2', 'av>=11,<19'))


def recognition_options(language, initial_prompt=''):
    """faster-whisper options for one pass.

    A glossary is passed as hotwords, which are prepended to every window.
    initial_prompt only biases the first window and then falls out of the
    context. Carrying it would require condition_on_previous_text to keep
    replaying earlier text, which is what spreads silence hallucinations.
    VAD and beam_size stay as they were.
    """
    options = dict(language=language, word_timestamps=True, vad_filter=True, beam_size=5)
    if initial_prompt:
        options['hotwords'] = initial_prompt
    return options


@app.function(image=image, gpu='L4', volumes={'/models': models}, timeout=3600)
def transcribe(audio, model, language, initial_prompt=''):
    """Yields the end time and words of each recognized segment."""
    import io
    from faster_whisper import WhisperModel
    recognizer = WhisperModel(model, device='cuda', compute_type='float16', download_root='/models')
    models.commit()
    segments, _ = recognizer.transcribe(io.BytesIO(audio), **recognition_options(language, initial_prompt))
    for segment in segments:
        yield {'end': segment.end, 'text': segment.text,
               'words': [{'text': word.word, 'start': word.start, 'end': word.end} for word in segment.words or []]}
