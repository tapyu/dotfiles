#!/bin/bash

# Unix shell script to bisynch iff an event has been done after 5min of the last modification 
# $1 -> drive path
# $2 -> local path

CURRENT_MD5=$(tar -C / -cf - "home/tapyu/books" | md5sum) # get md5 hash function of the tree directory
if [[ -f /home/tapyu/.cache/rclone/last_md5.log ]]; then
    if [[ ${CURRENT_MD5} != $(cat /home/tapyu/.cache/rclone/last_md5.log) ]]; then
        rclone bisync $1 $2
        echo ${CURRENT_MD5} > /home/tapyu/.cache/rclone/last_md5.log
    fi
else
    rclone bisync $1 $2
    echo ${CURRENT_MD5} > /home/tapyu/.cache/rclone/last_md5.log
fi