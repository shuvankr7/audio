import streamlit as st
import tempfile
import os
import sys

def check_dependencies():
    """Check if required packages are installed and install them if needed."""
    try:
        import whisper
    except ImportError:
        st.error("Whisper package not found. Please make sure it's installed: `pip install openai-whisper`")
        st.stop()

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using OpenAI Whisper with the small model.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    import whisper
    
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
    
    # Check dependencies
    check_dependencies()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/*")
        
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
                st.text_area("", transcription, height=300)
                
                # Add download button for the transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("If the error is related to GPU/CUDA, try running in CPU mode.")
        finally:
            # Remove the temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

if __name__ == "__main__":
    main()
