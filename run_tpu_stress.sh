#!/bin/bash

# Script to run TPU stress tests with different configurations

# Create profiles directory if it doesn't exist
mkdir -p profiles

# Function to run the stress test with specific parameters
run_test() {
    local matrix_size=$1
    local batch_size=$2
    local duration=$3
    local dtype=$4
    local profile=$5

    echo "========================================================"
    echo "Running TPU stress test with the following configuration:"
    echo "Matrix size: ${matrix_size}x${matrix_size}"
    echo "Batch size: $batch_size"
    echo "Duration: $duration seconds"
    echo "Data type: $dtype"
    echo "Profile: $profile"
    echo "========================================================"

    if [ "$profile" = "true" ]; then
        python tpu_stress_test.py --matrix_size $matrix_size --batch_size $batch_size --duration $duration --dtype $dtype --profile
    else
        python tpu_stress_test.py --matrix_size $matrix_size --batch_size $batch_size --duration $duration --dtype $dtype
    fi

    echo "Test completed."
    echo ""
}

# Default test - moderate load for 60 seconds
run_test 8192 4 60 bfloat16 false

# High intensity test - larger matrices for 30 seconds
run_test 16384 2 30 bfloat16 false

# Memory-intensive test - more batches of smaller matrices
run_test 4096 16 30 bfloat16 false

# Precision test - using float32 instead of bfloat16
run_test 8192 4 30 float32 false

# Profile run - shorter duration with profiling enabled
run_test 8192 4 15 bfloat16 true

echo "All tests completed!" 