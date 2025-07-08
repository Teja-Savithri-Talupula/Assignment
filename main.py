import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# Constants
VIDEO_PATHS = {
    'broadcast': 'videos/broadcast.mp4',
    'tacticam': 'videos/tacticam.mp4'
}
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PlayerReIDSystem:
    def __init__(self):
        # Feature extractor (ORB)
        self.feature_extractor = cv2.ORB_create()
        
        # Background subtractor for simple player detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Tracking variables
        self.player_db = defaultdict(list)  # {player_id: [features]}
        self.next_player_id = 1
        self.frame_count = 0
        self.similarity_threshold = 0.7

    def extract_frames(self, video_path, max_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
                
        cap.release()
        return frames

    def detect_players(self, frame):
        """Detect players using background subtraction"""
        fg_mask = self.back_sub.apply(frame)
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=4)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Minimum area threshold
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x+w, y+h])
            
        return boxes

    def extract_features(self, frame, boxes):
        """Extract ORB features for each detected player"""
        features = []
        crops = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
            
            # Detect and compute ORB features
            kp, des = self.feature_extractor.detectAndCompute(gray, None)
            
            if des is not None:
                # Use the first 32 features (pad if necessary)
                if len(des) > 32:
                    des = des[:32]
                else:
                    des = np.pad(des, ((0, 32 - len(des)), (0, 0)), mode='constant')
                
                features.append(des.flatten())  # Flatten to 1D array
                crops.append(player_crop)
            
        return features, crops

    def match_players(self, features, camera_id):
        """Match detected players to existing players or assign new IDs"""
        current_ids = []
        
        if not self.player_db:  # First frame, assign all new IDs
            for feature in features:
                player_id = self.next_player_id
                self.player_db[player_id].append(feature)
                current_ids.append(player_id)
                self.next_player_id += 1
            return current_ids
            
        # Prepare feature database
        db_ids = []
        db_features = []
        for pid, feats in self.player_db.items():
            db_ids.append(pid)
            db_features.append(feats[-1])  # Use most recent feature
            
        if not db_features:  # No players in DB yet
            for feature in features:
                player_id = self.next_player_id
                self.player_db[player_id].append(feature)
                current_ids.append(player_id)
                self.next_player_id += 1
            return current_ids
            
        # Convert to numpy arrays
        db_features = np.array(db_features)
        query_features = np.array(features)
        
        # Compute pairwise distances (smaller is better)
        distances = np.zeros((len(query_features), len(db_features)))
        for i, q_feat in enumerate(query_features):
            for j, db_feat in enumerate(db_features):
                distances[i,j] = np.linalg.norm(q_feat - db_feat)
        
        # Assign IDs based on smallest distance
        assigned_db_ids = set()
        current_ids = [-1] * len(features)
        
        while True:
            min_dist = np.min(distances)
            if min_dist > (1 - self.similarity_threshold) * 100:  # Convert similarity to distance threshold
                break
                
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            player_id = db_ids[j]
            
            if player_id not in assigned_db_ids and current_ids[i] == -1:
                current_ids[i] = player_id
                assigned_db_ids.add(player_id)
                self.player_db[player_id].append(features[i])
                
                # Keep only recent features
                if len(self.player_db[player_id]) > 10:
                    self.player_db[player_id] = self.player_db[player_id][-10:]
            
            # Set these to inf to exclude from next iterations
            distances[i, :] = np.inf
            distances[:, j] = np.inf
            
        # Assign new IDs to unmatched detections
        for i in range(len(current_ids)):
            if current_ids[i] == -1:
                player_id = self.next_player_id
                current_ids[i] = player_id
                self.player_db[player_id].append(features[i])
                self.next_player_id += 1
                
        return current_ids

    def visualize_results(self, frame, boxes, player_ids):
        """Draw bounding boxes and player IDs on frame"""
        for box, player_id in zip(boxes, player_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {player_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def process_videos(self, max_frames=None):
        """Main processing pipeline"""
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_broadcast = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, 'broadcast_annotated.mp4'),
            fourcc, 30, (1280, 720))
        out_tacticam = cv2.VideoWriter(
            os.path.join(OUTPUT_DIR, 'tacticam_annotated.mp4'),
            fourcc, 30, (1280, 720))
            
        # Process videos
        broadcast_frames = self.extract_frames(VIDEO_PATHS['broadcast'], max_frames)
        tacticam_frames = self.extract_frames(VIDEO_PATHS['tacticam'], max_frames)
        
        min_frames = min(len(broadcast_frames), len(tacticam_frames))
        
        for i in range(min_frames):
            self.frame_count = i
            
            # Process broadcast frame
            broadcast_frame = broadcast_frames[i]
            broadcast_boxes = self.detect_players(broadcast_frame)
            if broadcast_boxes:
                broadcast_features, _ = self.extract_features(broadcast_frame, broadcast_boxes)
                if broadcast_features:
                    broadcast_ids = self.match_players(broadcast_features, 'broadcast')
                    broadcast_frame = self.visualize_results(broadcast_frame, broadcast_boxes, broadcast_ids)
            
            # Process tacticam frame
            tacticam_frame = tacticam_frames[i]
            tacticam_boxes = self.detect_players(tacticam_frame)
            if tacticam_boxes:
                tacticam_features, _ = self.extract_features(tacticam_frame, tacticam_boxes)
                if tacticam_features:
                    tacticam_ids = self.match_players(tacticam_features, 'tacticam')
                    tacticam_frame = self.visualize_results(tacticam_frame, tacticam_boxes, tacticam_ids)
            
            # Write to output videos
            out_broadcast.write(broadcast_frame)
            out_tacticam.write(tacticam_frame)
            
            print(f'Processed frame {i+1}/{min_frames}')
            
        out_broadcast.release()
        out_tacticam.release()
        print("Processing complete. Results saved in 'outputs' directory.")

if __name__ == '__main__':
    system = PlayerReIDSystem()
    system.process_videos(max_frames=100)  # Set to None to process all frames
