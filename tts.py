import os
import re

import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoTokenizer, VitsModel

from config import TTS_MODEL

TTS_MODEL_NAME = TTS_MODEL

_tokenizer = None
_model = None


def _load_tts_backend():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
        _model = VitsModel.from_pretrained(TTS_MODEL_NAME)
        _model.eval()

    return _tokenizer, _model


def split_text_for_tts(text, max_characters=200):
    if len(text) <= max_characters:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_characters:
            words = sentence.split()
            for word in words:
                if len(word) > max_characters:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.extend(
                        word[index:index + max_characters]
                        for index in range(0, len(word), max_characters)
                    )
                    continue
                candidate = f"{current} {word}".strip()
                if len(candidate) > max_characters and current:
                    chunks.append(current)
                    current = word
                else:
                    current = candidate
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_characters and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def text_to_speech(
    text,
    output_path="tts_output.wav",
):

    if not text.strip():
        raise ValueError(
            "TTS text cannot be empty."
        )

    tokenizer, model = _load_tts_backend()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        waveform = model(**inputs).waveform

    samples = waveform.squeeze().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wavfile.write(output_path, model.config.sampling_rate, samples)

    return output_path


def text_to_speech_segments(text, output_directory="tts_segments"):
    output_directory = os.fspath(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    paths = []
    for index, chunk in enumerate(split_text_for_tts(text), start=1):
        paths.append(
            text_to_speech(
                chunk,
                os.path.join(output_directory, f"segment_{index}.wav"),
            )
        )
    return paths


def test_tts():

    text = (
        "Error thirty seven indicates "
        "an expiratory flow meter range error."
    )

    print("=" * 70)
    print("TTS TEST")
    print("=" * 70)

    print(f"Text: {text}")
    print()

    output_path = text_to_speech(
        text=text,
        output_path="test_tts.wav",
    )

    print(
        f"Audio saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    test_tts()