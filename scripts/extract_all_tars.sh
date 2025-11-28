#!/bin/bash

DATA_ROOT="${1:-/media/finn-avs1/DATA/Impromptu_VLA}"

echo "Extracting tar files from $DATA_ROOT"
echo "This may take a while"

find "$DATA_ROOT" -name "*.tar" -type f | while read -r tar_file; do
    tar_dir=$(dirname "$tar_file")

    echo "Extracting: $(basename "$tar_file")"
    tar -xf "$tar_file" -C "$tar_dir"
done

echo "Extration completed"
echo "You can now run: python scripts/data_organize.py --data_root $DATA_ROOT --split train"
echo "You can now run: python scripts/data_organize.py --data_root $DATA_ROOT --split val"
echo "After that you can run the format script: python scripts/format.py"
echo "Finally run: python scripts/merge_data.py"