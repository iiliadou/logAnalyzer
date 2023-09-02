#!/bin/bash

directory="C:\Users\ILI18009\Desktop\Thesis\Logs\Unstable-Builds"

for file in "$directory"/*; do
    if [[ -f "$file" ]]; then
        last_line=$(tail -n 1 "$file")
        if [[ "$last_line" == *"Finished: UNSTABLE"* ]]; then
            lines=$(cat "$file" | wc -l)
            head -n $((lines-1)) "$file" > "$file.tmp"
            mv "$file.tmp" "$file"
            echo "masked result"
        fi
    fi
done
