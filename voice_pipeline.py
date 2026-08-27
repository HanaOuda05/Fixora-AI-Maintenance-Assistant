
from pathlib import Path

from stt import transcribe_audio
from rag import answer_query
from tts import text_to_speech


BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "test_audios"
ANSWER_DIR = BASE_DIR / "answers"

ANSWER_DIR.mkdir(exist_ok=True)


def run_voice_pipeline(
    audio_path,
    device_id,
    output_audio="voice_answer.wav",
    generate_audio=True,
):

    print("=" * 70)
    print("VOICE PIPELINE")
    print("=" * 70)

    # 1. STT
    print("\n1. Transcribing audio...")

    query = transcribe_audio(audio_path)

    print(f"Transcription: {query}")

    # 2. RAG + LLM
    print("\n2. Generating answer...")

    result = answer_query(
        query=query,
        device_id=device_id,
    )

    display_text = result["display_text"]
    speech_text = result["speech_text"]

    print(f"Retrieval type: {result['retrieval_type']}")

    print()
    print("DISPLAY ANSWER:")
    print(display_text)

    print()
    print("SPEECH ANSWER:")
    print(speech_text)

    # 3. TTS
    audio_output = None

    if generate_audio:

        print("\n3. Converting spoken answer to speech...")

        audio_output = text_to_speech(
            text=speech_text,
            output_path=output_audio,
        )

        print(f"Audio saved to: {audio_output}")

    else:

        print("\n3. TTS skipped for testing.")

    return {
        "query": query,
        "display_text": display_text,
        "speech_text": speech_text,
        "retrieval_type": result["retrieval_type"],
        "audio_output": audio_output,
    }


def test_voice_pipeline():

    tests = [
        {
            "name": "Servo Error 37",
            "audio": AUDIO_DIR / "test_01_error37.m4a",
            "device_id": "servo_ventilator",
            "output": ANSWER_DIR / "answer_01_error37.wav",
        },
        {
            "name": "Monitor Blank Screen",
            "audio": AUDIO_DIR / "test_02_moniter.m4a",
            "device_id": "philips_g40",
            "output": ANSWER_DIR / "answer_02_monitor.wav",
        },
        {
            "name": "AC but not Battery",
            "audio": AUDIO_DIR / "test_03_AC.m4a",
            "device_id": "sc6002xl",
            "output": ANSWER_DIR / "answer_03_AC.wav",
        },
       {
    "name": "No CO2 Readings",
    "audio": AUDIO_DIR / "test_04_CO2.m4a",
    "device_id": "philips_g40",
    "output": ANSWER_DIR / "answer_04_CO2.wav",
},
        {
            "name": "G30 Blank Screen",
            "audio": AUDIO_DIR / "test_04_G30.m4a",
            "device_id": "philips_g40",
            "output": ANSWER_DIR / "answer_05_G30.wav",
        },
    ]

    print("=" * 70)
    print("FIXORA VOICE PIPELINE TEST SUITE")
    print("=" * 70)

    for index, test in enumerate(tests, start=1):

        print()
        print("#" * 70)
        print(f"TEST {index}: {test['name']}")
        print(f"DEVICE: {test['device_id']}")
        print(f"AUDIO: {test['audio']}")
        print("#" * 70)

        try:

            result = run_voice_pipeline(
                audio_path=test["audio"],
                device_id=test["device_id"],
                output_audio=test["output"],

                # IMPORTANT:
                # False = test STT + RAG only
                # True = also generate TTS audio
                generate_audio= True,
            )

            print()
            print("TEST RESULT")
            print("-" * 70)

            print(f"Question: {result['query']}")
            print(f"Retrieval: {result['retrieval_type']}")

            print()
            print("Display answer:")
            print(result["display_text"])

            print()
            print("Speech answer:")
            print(result["speech_text"])

            print()
            print("STATUS: PASSED")

        except Exception as error:

            print()
            print("STATUS: FAILED")
            print(f"ERROR: {error}")

    print()
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    test_voice_pipeline()