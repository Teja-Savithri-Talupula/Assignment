# Assignment
Player Re-identification System Code Implementation

Player Re-Identification System
Cross-Camera Player Tracking in Sports Footage


📌 Overview
This system detects and tracks players across two synchronized camera feeds (broadcast.mp4 and tacticam.mp4), assigning a consistent player_id to each player in both videos. It uses computer vision techniques (background subtraction + ORB features) for detection and re-identification.

🚀 Features
✅ Player Detection (Background Subtraction)
✅ Feature Extraction (ORB Descriptors)
✅ Cross-Camera ID Matching (Nearest Neighbor Search)
✅ Annotated Video Output (Bounding Boxes + Player IDs)

⚙️ Installation
1. Clone the Repository
bash
git clone https://github.com/yourusername/player_reid.git
cd player_reid
2. Install Dependencies
bash
pip install opencv-python opencv-contrib-python scikit-learn numpy
3. Prepare Video Files
Place broadcast.mp4 and tacticam.mp4 in the videos/ folder.

(Optional) Adjust video paths in player_reid.py if filenames differ.

🖥️ Usage
Run the Script
bash
python player_reid.py
Arguments:

Flag	Description	Default
--max_frames	Limit processed frames	100
--output_dir	Custom output directory	outputs/
Example:

bash
python player_reid.py --max_frames 500 --output_dir my_results/
Expected Output
Annotated videos saved in outputs/:

broadcast_annotated.mp4

tacticam_annotated.mp4

📂 Project Structure
text
player_reid/  
├── videos/                  # Input videos
│   ├── broadcast.mp4  
│   └── tacticam.mp4  
├── outputs/                 # Annotated results  
├── player_reid.py           # Main script  
└── README.md  
🔧 Customization
1. Adjust Detection Sensitivity
Modify in player_reid.py:

python
self.back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,       # Longer history = stabler background
    varThreshold=16,   # Higher = fewer detections (reduce noise)
    detectShadows=False
)
2. Change Feature Matching Threshold
python
self.similarity_threshold = 0.7  # Higher = stricter ID matching
⚠️ Troubleshooting
Issue	Solution
ModuleNotFoundError: No module named 'cv2'	Run pip install opencv-python opencv-contrib-python
No players detected	Increase varThreshold in background subtractor
Low performance	Reduce max_frames or video resolution
📜 License
MIT License - See LICENSE.
