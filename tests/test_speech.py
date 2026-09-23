
from speech.vad import VAD
from speech.asr import FasterWhisperBackend, IndicASRBackend, get_asr_backend
from speech.align import align_rationale_to_audio
from pipeline.analyze import analyze_audio
def test_vad():
    vad = VAD()
    assert isinstance(vad.is_available(), bool)

def test_asr_backends():
    fw = FasterWhisperBackend()
    assert fw.is_available()
    ia = IndicASRBackend()
    assert not ia.is_available()

def test_align():
    spans = align_rationale_to_audio([{"start_char": 0, "end_char": 1, "text": "a", "score": 0.5}], [], "a")
    assert spans[0]['audio_start'] == 0.0

def test_analyze_audio():
    res = analyze_audio("test.wav")
    assert res['input']['mode'] == 'audio'
