#!/bin/bash
# Copies 15 random files from a 'master' PDF directory to our local ML in-tray
# Note: this script only 'suggests' a copy command for you to inspect before
# executing it in your own shell session, i.e. "$ ./cp_random.sh | sh" if you
# trust the source directory contents.  
# ALWAYS better inspect the output, i.e. run cp_random.sh with no pipe, 
# then copy/paste the results yourself, to be sure!
#
find ~/Documents/PDF -maxdepth 1 -type f | shuf -n 15 | while IFS= read -r filename; do
    sanitized=$(basename "$filename" | sed 's/[^a-zA-Z0-9_.-]/_/g')
    echo cp -- "\"$filename\"" pdfs/$sanitized
done
