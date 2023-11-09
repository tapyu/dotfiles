#!/bin/sh

y_pos=$(( $(xdotool getactivewindow getwindowgeometry | \grep -oP 'Position: \d+,\K[^ ]+') > 0 ? 0 : 2160 ))
echo "The win y_pos is set to ${y_pos}" | tee /tmp/win-throw.log
echo "The win ID is $(xdotool getactivewindow)" | tee /tmp/win-throw.log
xdotool getactivewindow windowmove x ${y_pos}