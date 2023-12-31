# pose-classification
This repository contains the project related to the FVAB course. 

# About the project
Pose-emotion regression from a video is an emerging technology to identify and understand human emotions based on their body language and gestures. 
By analyzing the poses and movements (GAIT) of individuals captured in a video, this technology can infer their emotional states accurately.
By understanding users’ emotional states through their poses and gestures, systems and interfaces can be tailored to respond appropriately. 
Our work attempts to tackle this problem by means of a deep-learning architecture based on recurrent networks, specifically LSTM, to encode the poses of individuals extracted from the videos.
In addressing the problem, we frame it as a sequence-to-value regression task, mapping emotional states (happy, angry, sad, and neutral) to a 4-component vector. 
Two network architectures, EwalkNet and LandsAffectiveNet, are presented, each featuring a backbone encoder for landmarks and distinct affective features, results from experiments with these architectures are analyzed.

# How to use

The requirements used for the project are stored in the requirements.txt file.

0 - Create a directory named 'raw_video' inside the 'data' directory. 

1 - In the 'raw_video' subdirectory, insert the videos related to validation.

2 - In the 'data' directory, insert the Excel containing the dataset related to users' responses to the videos, and refactor the name of this file as "labels.xlsx".

3 - The main.py will be formatted as it follows (if you have refactored the Excel file as required in the step 2 you don't need to change anything in the following step):

      - FEATURES_CSV = os.path.join(DATA, "dataset.csv") # Automatically created (the name of the dataset containing the landmarks and the affective features extracted from the videos).
      
      - LABELS_XLSX  = os.path.join(DATA, "labels.xlsx") # Related to the Excel file refactored as labels.xlsx (the name of the Excel file containing the dataset related to users' responses to the videos).
      
      - LABELS_CSV   = os.path.join(DATA, "labels.csv") # Automatically created (the name of the CSV file containing the dataset related to users' responses to the videos preprocessed).
      

4 - Run the main.py to execute the pipeline, which consists of extracting features from the videos and saving them into a CSV, preprocessing the responses that constitute the labels for the regression task from the Excel file, and then providing them to the two main pre-trained architectures to plot the results."
