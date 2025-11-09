#!/bin/bash

# Path utama (pakai /d/ bukan /mnt/d/ karena ini di Git Bash)
BASE_DIR="/d/KULIAH/SEMESTER 5/Program Saint Data/Uranus/myfirstbook/Audio_recognition/Dataset_Voice_pertama"

# Loop untuk dua folder: Buka dan Tutup
for folder in "Buka" "Tutup"; do
    INPUT_DIR="$BASE_DIR/$folder"
    OUTPUT_DIR="$BASE_DIR/${folder}_wav"

    mkdir -p "$OUTPUT_DIR"
    echo "Mengonversi file dari $INPUT_DIR ke $OUTPUT_DIR ..."

    # Konversi semua file .aac
    for file in "$INPUT_DIR"/*.aac; do
        [ -e "$file" ] || { echo "Tidak ada file AAC di $INPUT_DIR"; continue; }
        filename=$(basename "$file" .aac)
        ffmpeg -y -i "$file" "$OUTPUT_DIR/${filename}.wav"
    done

    echo "Selesai untuk folder $folder."
    echo
done

echo "✅ Semua konversi selesai!"
