#!/bin/bash

# Batch Rollback script for scores.json files across all POPULATIONS
# Usage: ./rollback_scores.sh <backup_timestamp>
#
# This script:
# 1. Finds all scores.json files with backups matching the specified timestamp
# 2. Creates a backup of each current scores.json with current timestamp
# 3. Restores each scores.json from the specified backup

# Note: Not using 'set -e' here because arithmetic operations like ((var++))
# can return non-zero exit codes, which would terminate the script prematurely

# Check if backup timestamp/pattern is provided
if [ $# -eq 0 ]; then
    echo "Error: No backup timestamp pattern specified"
    echo "Usage: $0 <backup_pattern>"
    echo "Examples:"
    echo "  $0 20251017          # All backups from Oct 17, 2025"
    echo "  $0 20251017-13       # All backups from Oct 17, 2025, 1pm hour"
    echo "  $0 20251017-135503   # Specific backup timestamp"
    exit 1
fi

BACKUP_PATTERN="$1"
POPULATIONS_DIR="POPULATIONS"

# Check if POPULATIONS directory exists
if [ ! -d "$POPULATIONS_DIR" ]; then
    echo "Error: POPULATIONS directory not found"
    exit 1
fi

# Generate timestamp for new backups
NEW_TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "============================================"
echo "Batch Scores.json Rollback Script"
echo "============================================"
echo "Backup pattern: $BACKUP_PATTERN*"
echo "New backup timestamp: $NEW_TIMESTAMP"
echo "Searching in: $POPULATIONS_DIR"
echo ""

# Find all scores.json.bak files with the specified pattern
BACKUP_FILES=$(find "$POPULATIONS_DIR" -type f -name "scores.json.bak-${BACKUP_PATTERN}*" 2>/dev/null)

if [ -z "$BACKUP_FILES" ]; then
    echo "No backup files found matching pattern: $BACKUP_PATTERN*"
    exit 1
fi

# Count the number of backups found
BACKUP_COUNT=$(echo "$BACKUP_FILES" | wc -l)
echo "Found $BACKUP_COUNT backup file(s) to process"
echo ""

# Process each backup file
PROCESSED=0
SKIPPED=0
FAILED=0

while IFS= read -r BACKUP_FILE; do
    # Extract directory and construct current file path
    BACKUP_DIR=$(dirname "$BACKUP_FILE")
    CURRENT_FILE="$BACKUP_DIR/scores.json"
    
    echo "Processing: $CURRENT_FILE"
    
    # Check if current scores.json exists
    if [ ! -f "$CURRENT_FILE" ]; then
        echo "  ⚠ Warning: scores.json not found, skipping"
        SKIPPED=$((SKIPPED + 1))
        echo ""
        continue
    fi
    
    # Create new backup filename
    NEW_BACKUP="$CURRENT_FILE.bak-$NEW_TIMESTAMP"
    
    # Create backup of current scores.json
    if cp "$CURRENT_FILE" "$NEW_BACKUP" 2>/dev/null; then
        echo "  ✓ Current backed up to: $(basename "$NEW_BACKUP")"
    else
        echo "  ✗ Failed to backup current file"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi
    
    # Restore from specified backup
    if cp "$BACKUP_FILE" "$CURRENT_FILE" 2>/dev/null; then
        echo "  ✓ Restored from: $(basename "$BACKUP_FILE")"
        PROCESSED=$((PROCESSED + 1))
    else
        echo "  ✗ Failed to restore from backup"
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done <<< "$BACKUP_FILES"

echo "============================================"
echo "Rollback Summary"
echo "============================================"
echo "Successfully processed: $PROCESSED"
echo "Skipped (no current file): $SKIPPED"
echo "Failed: $FAILED"
echo ""

if [ $PROCESSED -gt 0 ]; then
    echo "Rollback completed successfully for $PROCESSED file(s)"
    echo "All current scores.json files backed up with timestamp: $NEW_TIMESTAMP"
    echo "All scores.json files restored from pattern: $BACKUP_PATTERN*"
else
    echo "No files were successfully processed"
    exit 1
fi
