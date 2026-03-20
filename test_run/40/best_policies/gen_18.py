import numpy as np
import math

def policy(state: list) -> float:
    state_arr = np.array(state).flatten()
    size = len(state_arr)
    
    # Indices for relevant features (every 7th element starting from index 0)
    throughput_idx = 0
    queue_idx = 1
    delay_idx = 2
    packet_loss_idx = 3
    jitter_idx = 4
    bandwidth_idx = 5
    error_rate_idx = 6

    throughput = state_arr[throughput_idx::7]
    queue = state_arr[queue_idx::7]
    delay = state_arr[delay_idx::7]
    packet_loss = state_arr[packet_loss_idx::7]
    jitter = state_arr[jitter_idx::7]
    bandwidth = state_arr[bandwidth_idx::7]
    error_rate = state_arr[error_rate_idx::7]
    
    # Calculate weighted features with optimized weights
    weight_throughput = 0.50
    weight_queue = -0.30
    weight_delay = -0.45
    weight_packet_loss = -0.55
    weight_jitter = -0.20
    weight_bandwidth = 0.35
    weight_error_rate = -0.45
    
    # Compute weighted sum
    weighted_sum = (
        weight_throughput * np.mean(throughput) +
        weight_queue * np.mean(queue) +
        weight_delay * np.mean(delay) +
        weight_packet_loss * np.mean(packet_loss) +
        weight_jitter * np.mean(jitter) +
        weight_bandwidth * np.mean(bandwidth) +
        weight_error_rate * np.mean(error_rate)
    )
    
    # Apply sigmoid activation for better scaling
    result = 1 / (1 + np.exp(-weighted_sum))
    
    # Scale to [-1, 1] using linear transformation
    scaled_result = 2 * (result - 0.5)
    
    return scaled_result