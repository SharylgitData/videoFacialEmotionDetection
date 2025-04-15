import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os


if not os.path.exists("recordings"):
    os.makedirs("recordings")

def main():
    st.title("Live Video Recorder")
    
    # Initialize session state variables
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'out' not in st.session_state:
        st.session_state.out = None
    if 'video_captured' not in st.session_state:
        st.session_state.video_captured = False
    if 'video_path' not in st.session_state:
        st.session_state.video_path = ""
    
    # Create a frame placeholder
    frame_placeholder = st.empty()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Buttons layout
    col1, col2 = st.columns(2)
    
    with col1:
        start_recording = st.button("Start Recording")
    
    with col2:
        stop_recording = st.button("Stop Recording")
    

    
    # Recording status
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        frame_placeholder.image(frame, channels="RGB")
        
        # Start recording
        if start_recording and not st.session_state.recording:
            st.session_state.recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.video_path = f"recordings/video_{timestamp}.mp4"
            
            # Get frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20.0
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            st.session_state.out = cv2.VideoWriter(
                st.session_state.video_path, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            status_text.success("Recording started...")
        
        # Stop recording
        if stop_recording and st.session_state.recording:
            st.session_state.recording = False
            if st.session_state.out is not None:
                st.session_state.out.release()
                st.session_state.out = None
                st.session_state.video_captured = True
                status_text.success("Recording stopped and saved")
        
        # Write frame to video if recording
        if st.session_state.recording and st.session_state.out is not None:
            # Convert back to BGR for writing
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            st.session_state.out.write(bgr_frame)
        
        # # Submit video
        # if submit_video and st.session_state.video_captured:
        #     if os.path.exists(st.session_state.video_path):
        #         st.session_state.video_captured = False
        #         status_text.success(f"Video submitted successfully! Saved at {st.session_state.video_path}")
        #         # Here you would typically add code to process the video
        #         # For example, upload to a server or process it further
        #     else:
        #         status_text.error("No video to submit. Please record first.")
        #     break
    
    # Release resources when done
    cap.release()
    if st.session_state.out is not None:
        st.session_state.out.release()

if __name__ == "__main__":
    main()