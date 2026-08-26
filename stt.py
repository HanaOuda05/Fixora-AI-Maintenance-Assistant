import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


STT_MODEL_NAME = "whisper-large-v3-turbo"


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def transcribe_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    with open(
        audio_path,
        "rb",
    ) as audio_file:

        transcription = (
            client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL_NAME,
                language="en",
                response_format="json",
                temperature=0.0,
            )
        )

    text = transcription.text.strip()

    return text


def test_stt():

    audio_path = r"C:\Users\Hanouna\Documents\Sound Recordings\Recording (9).m4a"
    print("STT TEST")
    print("=" * 70)

    print(
        f"Audio file: {audio_path}"
    )

    print()

    text = transcribe_audio(
        audio_path
    )

    print("TRANSCRIPTION:")
    print()

    print(text)


if __name__ == "__main__":
    test_stt()