#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

dir="$1"
label_scheme="$2"  # to be found in ocdi.labels.LABEL_SCHEME_TO_MAPPER

# Check if the directory exists
if [ ! -d "$dir" ]; then
    echo "Error: Directory '$dir' does not exist"
    exit 1
fi

# Run python script on all CSV files directly in the directory (not subdirectories)
for file in "$dir"/*.csv; do
    # Skip if no CSV files found
    if [ ! -f "$file" ]; then
        echo "No CSV files found in $dir"
        exit 1
    fi

    echo "Processing: $file"
    basename="${file%.csv}"
    output="${basename}-fasttext.txt"

    if ! python3 csv_to_fasttext.py "$file" "$label_scheme" > "$output"; then
        echo "Error: Python script failed for $file"
        exit 1
    fi
done

echo "Done!"
