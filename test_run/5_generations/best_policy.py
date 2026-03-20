import numpy as np
import math

def policy(state: list) -> float:
    """
    An improved congestion-control policy that uses weighted combination of features
    to produce a control action within [-1, 1], with optimized weights and scaling.
    
    The state is a flat list of 70 floats (10 timesteps × 7 features).
    Each feature represents different network metrics like throughput, loss rate, inverseRTT, etc.
    """
    # Convert the input list to a numpy array and flatten it (though it's already flat)
    state_array = np.array(state).flatten()
    
    # Ensure the state has the expected length
    if len(state_array) != 70:
        raise ValueError("State must contain exactly 70 elements")
    
    # Define weights for each feature (7 features × 10 timesteps)
    # Use optimized weights based on typical importance of each feature
    n_timesteps = len(state_array) // 7
    # Weighted importance for timesteps (more recent timesteps matter more)
    time_weights = np.linspace(0.8, 1.0, n_timesteps)  
    
    # Reshape the state array to (n_timesteps, n_features)
    state_2d = state_array.reshape((n_timesteps, 7))
    
    # Apply time weights to each timestep
    weighted_state = state_2d * time_weights[:, np.newaxis]
    
    # Extract each feature across all timesteps using flat indexing
    throughput = weighted_state[:, 0]          # Feature 0: throughput
    loss_rate = weighted_state[:, 1]          # Feature 1: loss rate
    inverse_rtt = weighted_state[:, 2]        # Feature 2: inverseRTT
    feature_3 = weighted_state[:, 3]           # Feature 3
    feature_4 = weighted_state[:, 4]           # Feature 4  
    feature_5 = weighted_state[:, 5]           # Feature 5
    feature_6 = weighted_state[:, 6]           # Feature 6
    
    # Optimized scoring function with feature-specific scaling and combining
    # Throughput: higher is better, scaled by typical range
    throughput_score = np.mean(throughput) * 0.6
    
    # Loss rate: lower is better, inverted and scaled 
    loss_score = -np.mean(loss_rate) * 0.9
    
    # Inverse RTT: higher is better (lower RTT), scaled appropriately
    rtt_score = np.mean(inverse_rtt) * 0.4
    
    # Additional features: combined with optimized weights
    extra_features = weighted_state[:, 3:]
    extra_score = np.mean(extra_features, axis=1)
    
    # Combine all scores with optimized weights based on empirical importance
    combined_score = (
        0.4 * throughput_score + 
        0.45 * loss_score + 
        0.1 * rtt_score +
        0.05 * np.mean(extra_score)
    )
    
    # Scale the combined score to [-1, 1] using tighter bounds based on empirical range
    # Assuming combined_score typically ranges from -1.8 to 2.2 after weighting
    scaled_output = 2 * (combined_score - (-1.8)) / (2.2 - (-1.8)) - 1
    
    # Clamp to ensure it's within [-1, 1] for robustness
    clamped_output = np.clip(scaled_output, -1.0, 1.0)
    
    return float(clamped_output)