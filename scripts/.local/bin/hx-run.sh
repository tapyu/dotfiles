#!/bin/bash

status_line=$(wezterm cli get-text | awk '$0 ~ /NOR/ { print $0 }')
filename=$(echo $status_line | awk '{ print $2}')
line_number=$(echo $status_line | awk '$0 ~ /NOR/ { split($NF, arr, ":"); print arr[1] }')
extension="${filename##*.}"

basedir=$(dirname "$filename")
basename=$(basename "$filename")
basename_without_extension="${basename%.*}"

case "$extension" in
  "c")
    pane_id=$(wezterm cli get-pane-direction down)
    if [ -z "${pane_id}"]; then
      pane_id=$(wezterm cli split-pane)
    fi

    run_command="clang -Wall -g -O3 $filename -o $basedir/$basename_without_extension && $basedir/$basename_without_extension"
    echo "${run_command}" | wezterm cli send-text --pane-id $pane_id --no-paste
    ;;
  "typ")
    typst compile $filename
    pdfname=${filename/%typ/pdf}
    if [ -z $(xdotool search --onlyvisible --name $pdfname) ]; then
      zathura $pdfname <&- & disown
    fi
    sleep 0.3 # wait to get the window done
    windowID=$(xdotool search --name $pdfname)
    echo $windowID
    echo $(wmctrl -l | grep $pdfname | awk '{ print $1 }')
    xdotool windowmove $windowID 1920 0
    xdotool windowsize $windowID 1920 2160
    ;;
esac

