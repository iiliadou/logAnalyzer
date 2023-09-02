#!/bin/bash

# set the directory you want to search
DIRECTORY="C:\Users\ILI18009\Desktop\Thesis\Logs\UnsortedMostlyFailed"
#DIRECTORY="\\\\le-n-885\Transfer\Logs\ipl\ipl-frameworks_s101-pipeline\builds"
# initialize counter variable
counter=328

# loop through each subdirectory in the directory
for subdirectory in "$DIRECTORY"/*/; do
  # check if the subdirectory contains a "log" file
  if [[ -f "${subdirectory}log" ]]; then
    # check if the last line contains "Finished: FAILURE"
     if grep -q "Finished: FAILURE" "${subdirectory}log"; then
      # copy the log file to a new directory with a new name
      new_filename="log${counter}"
      cp "${subdirectory}log" "C:\Users\ILI18009\Desktop\Thesis\Logs\Failed-Builds/${new_filename}"
      echo "Copied ${subdirectory}log to C:\Users\ILI18009\Desktop\Thesis\Logs\Failed-Builds/${new_filename}"
      # increment counter variable
      ((counter++))
    fi
  fi
done
