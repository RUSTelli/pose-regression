import os
import cv2
import numpy as np
from data_managing.consts import MAX_GAIT_LEN

def _frame_similarity(frame1, frame2) -> float:
    '''
    returns similarity score between two frames employing MSE.
    '''
    mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
    similarity = 1.0 / (1.0 + mse)
    return similarity

def _get_frame_similarities(video_path: str) -> list:
    '''
    returns the list of frame similarities between the first 
    and each frame of the video. 
    '''
    video = cv2.VideoCapture(video_path)
    
    # Read the first frame
    _, first_frame = video.read()
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    frame_similarities = []
    frame_num = 0
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            # Reached the end of the video
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        similarity = _frame_similarity(first_frame_gray, frame_gray)
        
        frame_similarities.append(similarity)
        frame_num += 1
    
    # Release the video file
    video.release()
    
    return frame_similarities

def _same_magnitude(num1: float, num2: float) -> bool:
    '''
    checks if two number are the same order of scale
    '''
    exp1 = int(f"{num1:.0e}".split("e")[1])
    exp2 = int(f"{num2:.0e}".split("e")[1])
    
    return exp1 == exp2

def _gait_cycle_frames_count(video_path: str) -> int:
    '''
    return the number of frames in a single gait cycle
    '''
    similarities = _get_frame_similarities(video_path)
    previous_similarity = similarities[0]
    frame_count = 0

    for similarity in similarities:
        frame_count += 1
        if similarity > previous_similarity:
            if not _same_magnitude(similarity, previous_similarity):
                # It means that the current frame is way more similar to the  
                # first one rather than the previous one: -> gait clip starts repeating 
                return frame_count
        previous_similarity = similarity

def _extract_gait_frames(video_path: str) -> list:
    '''
    return the gait cycle's frames.
    '''
    frames = []
    video = cv2.VideoCapture(video_path)
    n_frames = _gait_cycle_frames_count(video_path) 

    # Read until the specified number of frames is reached
    for _ in range(n_frames):
        ret, frame = video.read()
        frames.append(frame)
    
    # Release the video file and return the frames
    video.release()
    return frames if n_frames <= MAX_GAIT_LEN else frames[:MAX_GAIT_LEN]

def _save_video_from_frames(frames: list, output_file: str) -> None:
    '''
    saves a list of frames as a .mp4 video to the specified output destination.
    '''
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames_no = len(frames)
    out = cv2.VideoWriter(output_file, fourcc, frames_no, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_file}")

def extract_gaits_from_folder(unprocessed: str, processed: str) -> None:
    # Create the processed folder if it doesn't exist
    if not os.path.exists(processed):
        os.makedirs(processed)
    for filename in os.listdir(unprocessed):
            if filename.endswith(".mp4"):
                raw_video_path       = os.path.join(unprocessed, filename)
                processed_video_path = os.path.join(processed, filename)
                # store single gait cycle video
                _save_video_from_frames(
                    frames = _extract_gait_frames(raw_video_path),
                    output_file = processed_video_path
                )