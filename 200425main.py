import streamlit as st
import tempfile
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
import os
from datetime import datetime

# Set up directories
if not os.path.exists("recordings"):
    os.makedirs("recordings")

# Page configuration
st.set_page_config(page_title="Emotion Detection Recorder", layout="wide")
st.title("🎥 Vedio Facial Emotion Detector ")

@st.cache_resource
def load_models():
    device = "cpu"
    model_name = get_model_list()[0]
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
    return fer, device

fer, device = load_models()

def recognize_faces(frame: np.ndarray):
    try:
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        return bounding_boxes[probs > 0.9]
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        return []

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    sample_frames = []
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        st.error("Invalid video file or unable to read frame count")
        return None, None
    
    processed_frames = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bounding_boxes = recognize_faces(image_rgb)
        
        facial_images = []
        for bbox in bounding_boxes:
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = image_rgb[y1:y2, x1:x2, :]
            if face_img.size > 0:
                facial_images.append(face_img)
        
        if facial_images:
            try:
                emotions, scores = fer.predict_emotions(facial_images, logits=True)
                results.extend(scores)
                if len(sample_frames) < 5:
                    sample_frames.extend(zip(facial_images, emotions))
            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
        
        processed_frames += 1
        progress_bar.progress(min(processed_frames / total_frames, 1.0))
    
    cap.release()
    progress_bar.empty()
    
    if results:
        avg_scores = np.mean(results, axis=0)
        emotion_idx = np.argmax(avg_scores)
        return fer.idx_to_emotion_class[emotion_idx], sample_frames[:5]
    return "No faces detected", []

def analyze_recorded_video(video_path):
    if os.path.exists(video_path):
        with st.spinner('Analyzing video...'):
            dominant_emotion, sample_frames = process_video(video_path)
            
            if dominant_emotion is None:
                st.error("Failed to process video")
            else:
                st.success("Analysis complete!")
                # st.subheader(f"Dominant Emotion: {dominant_emotion}")
                
                if sample_frames:
                    st.subheader("Sample Detections")
                    cols = st.columns(5)
                    for idx, (img, emotion) in enumerate(sample_frames):
                        with cols[idx]:
                            st.image(img, caption=emotion, clamp=True)
                else:
                    st.warning("No faces detected in the video")

def main():
   
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'out' not in st.session_state:
        st.session_state.out = None
    if 'video_captured' not in st.session_state:
        st.session_state.video_captured = False
    if 'video_path' not in st.session_state:
        st.session_state.video_path = ""
    
   

    
    # with tab1:
    #     st.header("Live Emotion Recording")

    #             # buttons layout
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         start_recording = st.button("Start Recording", key="start_recording")
        
    #     with col2:
    #         stop_recording = st.button("Stop Recording", key="stop_recording")


        
  
    #     try:
    #         cap = cv2.VideoCapture(0)
    #         if not cap.isOpened():
    #             st.error("""
    #             Could not access camera. Please:
    #             1. Check if your camera is connected
    #             2. Ensure no other applications are using the camera
    #             3. Grant camera permissions to your browser
    #             """)
    #             st.stop()
    #     except Exception as e:
    #         st.error(f"Camera initialization error: {str(e)}")
    #         st.stop()
        
    #     # create a frame placeholder
    #     frame_placeholder = st.empty()
        

        
    #     # recording status
    #     status_text = st.empty()
        
    #     try:
    #         while cap.isOpened():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 status_text.error("Failed to capture video frame")
    #                 break
                
    #             #convert frame from BGR to RGB
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
    #             # display the frame
    #             frame_placeholder.image(frame, channels="RGB")
                
    #             # start recording
    #             if start_recording and not st.session_state.recording:
    #                 st.session_state.recording = True
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 st.session_state.video_path = f"recordings/video_{timestamp}.mp4"
                    
    #                 # get frame dimensions
    #                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                 fps = 20.0
                    
    #                 # initialize video writer
    #                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #                 st.session_state.out = cv2.VideoWriter(
    #                     st.session_state.video_path, 
    #                     fourcc, 
    #                     fps, 
    #                     (frame_width, frame_height)
    #                 )
    #                 status_text.success("Recording started...")
                
    #             # stop recording
    #             if stop_recording and st.session_state.recording:
    #                 st.session_state.recording = False
    #                 if st.session_state.out is not None:
    #                     st.session_state.out.release()
    #                     st.session_state.out = None
    #                     st.session_state.video_captured = True
    #                     status_text.success("Recording stopped and saved")
    #                     break
                
    #             # write frame to video if recording
    #             if st.session_state.recording and st.session_state.out is not None:
    #                 # Convert back to BGR for writing
    #                 bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #                 st.session_state.out.write(bgr_frame)
        
    #     finally:
    #         # release resources when done
    #         cap.release()
    #         if st.session_state.out is not None:
    #             st.session_state.out.release()
        
    #     # analyze the recorded video if one was captured
    #     if st.session_state.video_captured and st.session_state.video_path:
    #         st.video(st.session_state.video_path)
    #         analyze_recorded_video(st.session_state.video_path)
    

    st.header("Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
    
    if uploaded_file is not None:
        # save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        
        # show video preview
        st.video(temp_path)
        
        # process video
        analyze_recorded_video(temp_path)
        
        # clean up temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            st.warning(f"Could not delete temp file: {str(e)}")

if __name__ == "__main__":
    main()