import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


TTS_MODEL_NAME = "canopylabs/orpheus-v1-english"
TTS_VOICE = "hannah"


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def text_to_speech(
    text,
    output_path="tts_output.wav",
):

    if not text.strip():
        raise ValueError(
            "TTS text cannot be empty."
        )

    response = client.audio.speech.create(
        model=TTS_MODEL_NAME,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )

    response.write_to_file(
        output_path
    )

    return output_path


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