import os
import tempfile
from pathlib import Path

import streamlit as st

from rag import answer_query
from stt import transcribe_audio
from tts import text_to_speech


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fixora",
    page_icon="🔧",
    layout="wide",
)


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

DEVICE_MAP = {
    "Servo Ventilator System": "servo_ventilator",
    "Philips G30/G40 Patient Monitor": "philips_g40",
    "SC 6002XL Patient Monitor": "sc6002xl",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_audio(uploaded_file):
    """
    Save Streamlit uploaded/recorded audio to a temporary file
    so the STT module can process it.
    """

    suffix = Path(uploaded_file.name).suffix

    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        return temp_file.name


def generate_tts_audio(text):
    """
    Generate a temporary WAV response using the local TTS model.
    """

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    )

    temp_audio.close()

    text_to_speech(
        text=text,
        output_path=temp_audio.name,
    )

    return temp_audio.name


# ============================================================
# SESSION STATE
# ============================================================

if "query" not in st.session_state:
    st.session_state.query = ""

if "display_text" not in st.session_state:
    st.session_state.display_text = ""

if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""

if "retrieval_type" not in st.session_state:
    st.session_state.retrieval_type = ""

if "answer_audio" not in st.session_state:
    st.session_state.answer_audio = None


# ============================================================
# HEADER
# ============================================================

st.title("Fixora")

st.subheader(
    "AI-Powered Maintenance Troubleshooting Assistant"
)

st.write(
    "Select a device, describe the problem using text or voice, "
    "and Fixora will retrieve relevant service-manual evidence "
    "and generate a grounded troubleshooting response."
)

st.divider()


# ============================================================
# DEVICE SELECTION
# ============================================================

st.subheader("1. Select Device")

selected_device = st.selectbox(
    "Device / Model",
    list(DEVICE_MAP.keys()),
)

device_id = DEVICE_MAP[selected_device]

st.caption(
    f"Selected knowledge base: {selected_device}"
)

st.divider()


# ============================================================
# INPUT METHOD
# ============================================================

st.subheader("2. Describe the Problem")

input_method = st.radio(
    "Choose input method",
    [
        "Text",
        "Voice",
    ],
    horizontal=True,
)


# ============================================================
# TEXT INPUT
# ============================================================

query = None


if input_method == "Text":

    text_query = st.text_area(
        "Type your maintenance question",
        placeholder=(
            "Example: The monitor turns on "
            "but the screen is blank."
        ),
        height=120,
    )

    if st.button(
        "Analyze Problem",
        type="primary",
        use_container_width=True,
    ):

        if not text_query.strip():

            st.warning(
                "Please enter a maintenance question."
            )

        else:

            query = text_query.strip()


# ============================================================
# VOICE INPUT
# ============================================================

else:

    voice_option = st.radio(
        "Voice source",
        [
            "Record microphone",
            "Upload audio file",
        ],
        horizontal=True,
    )

    audio_file = None

    if voice_option == "Record microphone":

        audio_file = st.audio_input(
            "Record your maintenance question"
        )

    else:

        audio_file = st.file_uploader(
            "Upload an audio recording",
            type=[
                "wav",
                "mp3",
                "m4a",
                "mp4",
                "webm",
                "ogg",
            ],
        )

    if audio_file is not None:

        st.audio(audio_file)

        if st.button(
            "Transcribe and Analyze",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Transcribing your audio..."
                ):

                    audio_path = save_uploaded_audio(
                        audio_file
                    )

                    query = transcribe_audio(
                        audio_path
                    )

                st.session_state.query = query

            except Exception as error:

                st.error(
                    f"Speech-to-text failed: {error}"
                )


# ============================================================
# RUN RAG
# ============================================================

if query:

    st.session_state.query = query

    try:

        with st.spinner(
            "Searching the service manual and generating an answer..."
        ):

            result = answer_query(
                query=query,
                device_id=device_id,
            )

        st.session_state.display_text = (
            result["display_text"]
        )

        st.session_state.speech_text = (
            result["speech_text"]
        )

        st.session_state.retrieval_type = (
            result["retrieval_type"]
        )

        # Remove previous audio when new answer generated
        st.session_state.answer_audio = None

    except Exception as error:

        st.error(
            f"Fixora could not generate an answer: {error}"
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.query:

    st.divider()

    st.subheader("3. Fixora Analysis")

    st.markdown("**Question**")

    st.info(
        st.session_state.query
    )


if st.session_state.display_text:

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        st.markdown(
            "**Manual-grounded answer**"
        )

    with col2:

        retrieval = (
            st.session_state.retrieval_type
        )

        if retrieval == "exact_error":

            st.success(
                "Exact error match"
            )

        else:

            st.info(
                "Semantic retrieval"
            )

    st.markdown(
        st.session_state.display_text
    )


# ============================================================
# TTS
# ============================================================

if st.session_state.speech_text:

    st.divider()

    st.subheader("4. Voice Response")

    st.markdown(
        "**Speech version**"
    )

    st.write(
        st.session_state.speech_text
    )

    if st.button(
        "Generate Voice Answer",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Generating speech locally..."
            ):

                audio_path = generate_tts_audio(
                    st.session_state.speech_text
                )

                st.session_state.answer_audio = (
                    audio_path
                )

        except Exception as error:

            st.error(
                f"Text-to-speech failed: {error}"
            )


# ============================================================
# AUDIO PLAYBACK + DOWNLOAD
# ============================================================

if st.session_state.answer_audio:

    audio_path = (
        st.session_state.answer_audio
    )

    st.audio(
        audio_path,
        format="audio/wav",
    )

    with open(
        audio_path,
        "rb",
    ) as audio_file:

        audio_bytes = audio_file.read()

    st.download_button(
        label="Download Voice Answer",
        data=audio_bytes,
        file_name="fixora_answer.wav",
        mime="audio/wav",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fixora answers are generated from the selected "
    "service manual evidence. Verify maintenance actions "
    "against the manufacturer's documentation before execution."
)