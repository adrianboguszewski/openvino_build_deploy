import numpy as np


class AssociativeEmbeddingDecoder:
    """Decoder for associative embedding pose estimation models like human-pose-estimation-0005.
    
    Uses heatmaps to detect keypoints and embeddings to group them into person instances.
    """
    
    def __init__(self, num_joints=17, max_num_people=30, detection_threshold=0.1, 
                 tag_threshold=1.0, use_detection_val=True, ignore_too_much=False):
        """Initialize the decoder.
        
        Args:
            num_joints: Number of keypoints (17 for COCO format)
            max_num_people: Maximum number of people to detect
            detection_threshold: Threshold for keypoint detection
            tag_threshold: Threshold for grouping keypoints by embedding distance
            use_detection_val: Whether to use detection scores in final score
            ignore_too_much: Whether to ignore detections with too many people
        """
        self.num_joints = num_joints
        self.max_num_people = max_num_people
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much
    
    def __call__(self, heatmaps, embeddings):
        """Decode poses from heatmaps and embeddings.
        
        Args:
            heatmaps: (1, 17, H, W) - keypoint heatmaps
            embeddings: (1, 17, H, W, 1) - associative embeddings for grouping
            
        Returns:
            poses: (N, 17, 3) array of poses, each with 17 keypoints (x, y, score)
            scores: (N,) array of pose confidence scores
        """
        batch_size = heatmaps.shape[0]
        assert batch_size == 1, 'Batch size of 1 only supported'
        
        # Remove batch dimension and handle embedding shape
        heatmaps = heatmaps[0]  # (17, H, W)
        embeddings = embeddings[0]  # (17, H, W, 1) or (17, H, W)
        if embeddings.ndim == 4:
            embeddings = embeddings.squeeze(-1)  # (17, H, W)
        
        # Get all detected keypoints with their embeddings
        all_keypoints, all_tags = self._extract_keypoints_with_tags(heatmaps, embeddings)
        
        # Group keypoints into person instances using embeddings
        poses, scores = self._group_keypoints_by_tags(all_keypoints, all_tags)
        
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
            scores = np.asarray(scores, dtype=np.float32)
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
        
        return poses, scores
    
    def _extract_keypoints_with_tags(self, heatmaps, embeddings):
        """Extract keypoints and their embedding tags from heatmaps.
        
        Args:
            heatmaps: (17, H, W) - keypoint heatmaps
            embeddings: (17, H, W) - embedding tags
            
        Returns:
            all_keypoints: List of arrays, one per joint type
            all_tags: List of arrays, one per joint type
        """
        num_joints, height, width = heatmaps.shape
        
        all_keypoints = []
        all_tags = []
        
        for joint_idx in range(num_joints):
            heatmap = heatmaps[joint_idx]
            embedding = embeddings[joint_idx]
            
            # Find local maxima in heatmap
            keypoints, tags = self._get_keypoints_from_heatmap(
                heatmap, embedding, self.detection_threshold
            )
            
            all_keypoints.append(keypoints)
            all_tags.append(tags)
        
        return all_keypoints, all_tags
    
    def _get_keypoints_from_heatmap(self, heatmap, embedding, threshold):
        """Extract keypoints from a single joint heatmap.
        
        Args:
            heatmap: (H, W) - heatmap for one joint
            embedding: (H, W) - embeddings for one joint
            threshold: detection threshold
            
        Returns:
            keypoints: (N, 3) array of (x, y, score)
            tags: (N,) array of embedding values
        """
        # Find all points above threshold
        mask = heatmap > threshold
        
        if not mask.any():
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        
        # Get coordinates and scores
        ys, xs = np.where(mask)
        scores = heatmap[mask]
        embeddings_vals = embedding[mask]
        
        # Create keypoints array
        keypoints = np.stack([xs, ys, scores], axis=1).astype(np.float32)
        tags = embeddings_vals.astype(np.float32)
        
        return keypoints, tags
    
    def _group_keypoints_by_tags(self, all_keypoints, all_tags):
        """Group keypoints into person instances based on embedding similarity.
        
        Args:
            all_keypoints: List of (N_i, 3) arrays for each joint
            all_tags: List of (N_i,) arrays for each joint
            
        Returns:
            poses: List of pose arrays
            scores: List of pose scores
        """
        # Find valid keypoints (those detected)
        joint_order = []
        tags_list = []
        keypoints_list = []
        
        for joint_idx in range(self.num_joints):
            if len(all_keypoints[joint_idx]) > 0:
                joint_order.append(joint_idx)
                tags_list.append(all_tags[joint_idx])
                keypoints_list.append(all_keypoints[joint_idx])
        
        if len(joint_order) == 0:
            return [], []
        
        # Start with keypoints from first detected joint
        poses = []
        pose_tags = []
        pose_scores = []
        
        # Use a simple grouping strategy: match keypoints with similar embedding values
        # Process each keypoint from the first joint type as a potential person
        first_joint_idx = joint_order[0]
        
        for kpt_idx in range(len(keypoints_list[0])):
            pose = np.zeros((self.num_joints, 3), dtype=np.float32)
            pose[first_joint_idx] = keypoints_list[0][kpt_idx]
            tag = tags_list[0][kpt_idx]
            
            # Find matching keypoints in other joints
            num_matched = 1
            total_score = keypoints_list[0][kpt_idx, 2]
            
            for i in range(1, len(joint_order)):
                joint_idx = joint_order[i]
                
                if len(tags_list[i]) == 0:
                    continue
                
                # Find keypoint with closest embedding value
                tag_diffs = np.abs(tags_list[i] - tag)
                min_idx = np.argmin(tag_diffs)
                min_diff = tag_diffs[min_idx]
                
                # If embedding is close enough, assign this keypoint to the pose
                if min_diff < self.tag_threshold:
                    pose[joint_idx] = keypoints_list[i][min_idx]
                    num_matched += 1
                    total_score += keypoints_list[i][min_idx, 2]
            
            # Only keep poses with enough keypoints
            if num_matched >= 3:  # At least 3 keypoints
                poses.append(pose.flatten())
                avg_score = total_score / max(num_matched, 1)
                pose_scores.append(avg_score * num_matched)  # Score weighted by keypoints
        
        # Filter to max number of people
        if len(poses) > self.max_num_people:
            # Keep top scoring poses
            indices = np.argsort(pose_scores)[::-1][:self.max_num_people]
            poses = [poses[i] for i in indices]
            pose_scores = [pose_scores[i] for i in indices]
        
        return poses, pose_scores
