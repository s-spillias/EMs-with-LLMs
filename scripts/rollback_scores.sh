#!/bin/bash

# Rollback script for scores.json
# Usage: ./rollback_scores.sh <path_to_backup_file>
#
# This script:
# 1. Creates a backup of the current scores.json with current timestamp
# 2. Restores scores.json from the specified backup file

set -e  # Exit on error

# Check if backup file path is provided
if [ $# -eq 0 ]; then
    echo "Error: No backup file specified"
    echo "Usage: $0 <path_to_backup_file>"
    echo "Example: $0 POPULATIONS/POPULATION_0009/INDIVIDUAL_A5QU60KP/scores.json.bak-20251017-135503"
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Extract directory and filename from backup file path
BACKUP_DIR=$(dirname "$BACKUP_FILE")
CURRENT_FILE="$BACKUP_DIR/scores.json"

# Check if current scores.json exists
if [ ! -f "$CURRENT_FILE" ]; then
    echo "Error: Current scores.json not found: $CURRENT_FILE"
    exit 1
fi

# Generate timestamp for current backup
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
NEW_BACKUP="$CURRENT_FILE.bak-$TIMESTAMP"

echo "============================================"
echo "Scores.json Rollback Script"
echo "============================================"
echo "Current file: $CURRENT_FILE"
echo "Backup source: $BACKUP_FILE"
echo "Saving current to: $NEW_BACKUP"
echo ""

# Create backup of current scores.json
echo "Step 1: Creating backup of current scores.json..."
cp "$CURRENT_FILE" "$NEW_BACKUP"
echo "✓ Current scores.json backed up to: $NEW_BACKUP"
echo ""

# Restore from specified backup
echo "Step 2: Restoring scores.json from backup..."
cp "$BACKUP_FILE" "$CURRENT_FILE"
echo "✓ scores.json restored from: $BACKUP_FILE"
echo ""

echo "============================================"
echo "Rollback completed successfully!"
echo "============================================"
echo "Summary:"
echo "  - Current scores.json backed up to: $NEW_BACKUP"
echo "  - scores.json restored from: $BACKUP_FILE"
echo ""
