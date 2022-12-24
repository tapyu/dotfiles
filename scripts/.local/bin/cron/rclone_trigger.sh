#!/bin/bash

# Unix shell script to bisynch iff a event has been done after 5min of the last modification 
# $1 -> drive path
# $2 -> local path
# $3 -> remote path

LAST_SYNC_DATE=(`date`)
inotifywait --quiet --monitor --recursive --event modify,delete,create $1 | while read DIRECTORY EVENT FILE; do
  DATE=(`date`)
  TIME_DIFF = $(( (${LAST_SYNC[4]:0:2} - ${DATE[4]:0:2})*60 + LAST_SYNC[4]:3:2} - ${DATE[4]:3:2))
  if [[ ${LAST_SYNC_DATE[1]} != ${DATE[1]} ]] || [[ $TIME_DIFF -gt 5 ]]; then
    rclone bisync $2 $3
    LAST_SYNC_DATE=$DATE
  fi
do