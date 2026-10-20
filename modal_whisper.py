"""faster-whisper on a Modal GPU, called by scripter.py's --engine modal."""
import modal

app = modal.App('scripter-whisper')
models = modal.Volume.from_name('scripter-whisper-models', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04', add_python='3.11')
         .pip_install('faster-whisper>=1.2,<2', 'av>=11,<19'))


@app.function(image=image, gpu='L4', volumes={'/models': models}, timeout=3600)
def transcribe(audio, model, language):
    """Yields the end time and words of each recognized segment."""
    import io
    from faster_whisper import WhisperModel
    recognizer = WhisperModel(model, device='cuda', compute_type='float16', download_root='/models')
    models.commit()
    segments, _ = recognizer.transcribe(io.BytesIO(audio), language=language, word_timestamps=True,
                                        vad_filter=True, beam_size=5)
    for segment in segments:
        yield {'end': segment.end, 'text': segment.text,
               'words': [{'text': word.word, 'start': word.start, 'end': word.end} for word in segment.words or []]}
