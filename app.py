import streamlit as st
import whisper
import tempfile
import os

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using OpenAI Whisper with the small model.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    # Load the small model
    with st.spinner('Loading Whisper model...'):
        model = whisper.load_model("small")
    
    # Transcribe the audio
    with st.spinner('Transcribing audio...'):
        result = model.transcribe(audio_file_path)
    
    return result["text"]

def main():
    st.title("Audio Transcription with Whisper")
    st.write("Upload an audio file to transcribe it using OpenAI's Whisper model.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Add a transcribe button
            if st.button('Transcribe Audio'):
                # Transcribe the audio
                transcription = transcribe_audio(tmp_file_path)
                
                # Display the result
                st.subheader("Transcription Result")
                st.write(transcription)
                
                # Add download button for the transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Remove the temporary file
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
