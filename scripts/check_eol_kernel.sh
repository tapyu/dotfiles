#!/bin/env bash
# https://forum.manjaro.org/t/can-i-get-a-warning-about-eol-of-a-kernel/84079/10

# check_eol_kernel
# Check if one of the current installed Linux kernels has reached EOL
#
# Usage:
#   check_eol_kernel
#   check_eol_kernel list
#
# Compares the installed Linux kernel versions to the current supported kernels
# from Manjaro.  If one of the installed kernels are not supported anymore,
# meaning they are marked as EOL (end of life) in Manjaro, then a warning
# message is printed out.
# 
# It is achieving this by downloading official announcement news and strips out
# the relevant list of kernels.  This will be compared to the list of currently
# installed kernels reported by `mhwd-kernel` tool from Manjaro.
#
# list:
#   If this script is called with an argument "list", then it will not check
#   current installed versions.  Instead it will just print out the list of EOL
#   marked kernels retrieved from the news page.
# 

declare -a eol=( \
    $(curl -Ls "https://forum.manjaro.org/c/announcements/stable-updates.rss" \
    | awk -F'>| ' '/\[EOL\]/ {print " "$2}' \
    | sort | uniq | grep -Eo "(linux[0-9]+ *)+")
)
# eol=(linux511 linux512 linux513 linux57 linux58 linux59)    # result today

if [[ "$1" == "list" ]]
then
    echo "${eol[*]}"
    exit
fi

declare -a installed=($(mhwd-kernel -li | awk '/* / {print $2}'))
# installed=(linux510 linux512 linux514)    # result example

for k in "${installed[@]}"; do
    if [[ $(printf "%s\n" "${eol[@]}"|grep ^$k$ -c) > 0 ]]; then
        # oops linux512 is in EOL list
        echo "Warning: kernel \"$k\" installed but is EOL"
        #exit 5  # exit in hook not break pacman transaction but is big error for user
    fi
done