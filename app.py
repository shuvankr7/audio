import streamlit as st
import tempfile
import os
import sys

# Load whisper model at startup
@st.cache_resource
def load_whisper_model():
    try:
        import whisper
        return whisper.load_model("small")
    except ImportError:
        st.error("Whisper module not found. Please ensure it's installed correctly.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        st.stop()

def main():
    st.title("Audio Transcription with Whisper")
    st.write("Upload an audio file to transcribe it using OpenAI's Whisper model.")
    
    # Load model at startup
    with st.spinner("Loading Whisper model... This may take a moment."):
        model = load_whisper_model()
    
    st.success("Whisper model loaded and ready!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format="audio/*")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Transcribe button
        if st.button('Transcribe Audio'):
            try:
                with st.spinner("Transcribing audio..."):
                    try:
                        # Use the already loaded model
                        result = model.transcribe(tmp_file_path)
                        transcription = result["text"]
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
                        st.stop()
                
                # Display the transcription
                st.subheader("Transcription Result")
                st.text_area("", transcription, height=300)
                
                # Download button
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

if __name__ == "__main__":
    main()
