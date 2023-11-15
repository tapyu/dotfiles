#!/bin/sh

second_screen=$(xrandr | grep -P '\bconnected' | grep -o '\bDP-[[:digit:]]')
main_screen=$(xrandr | grep -o 'eDP-[[:digit:]]-[[:digit:]]')

xrandr --output $second_screen --scale 2x2
xrandr --output $main_screen --pos 0x2160
xrandr --output $second_screen --pos 0x0
