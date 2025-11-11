#!/bin/bash
# Railway build script - trains models before deployment

set -e  # Exit on error

echo "========================================="
echo "Starting ACV Model Training Build"
echo "========================================="

# Check if data exists
if [ ! -f "used_cars (1).csv" ]; then
    echo "ERROR: used_cars (1).csv not found!"
    echo "Data file must be present for model training"
    exit 1
fi

# Create models directory
mkdir -p models

# Train the models
echo "Training production models..."
python3 train.py

# Verify models were created
if [ ! -f "models/production_tree_model.pkl" ]; then
    echo "ERROR: Model training failed - production_tree_model.pkl not found"
    exit 1
fi

echo "✓ Model training complete"
echo "✓ Models saved to models/ directory"

# List created models
echo ""
echo "Created models:"
ls -lh models/

echo "========================================="
echo "Build Complete - Ready to Deploy"
echo "========================================="
