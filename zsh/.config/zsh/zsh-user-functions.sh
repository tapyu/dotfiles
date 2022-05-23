# Use ranger to switch directories and bind it to ctrl-o
rangercd () {
    tmp="$(mktemp)"
    ranger --choosedir="$tmp" "$@"
    if [ -f "$tmp" ]; then
        dir="$(cat "$tmp")"
        rm -f "$tmp"
        [ -d "$dir" ] && [ "$dir" != "$(pwd)" ] && cd "$dir"                                               
    fi
}
bindkey -s '^o' 'rangercd\n'


# function that uses youtube-dl along with ffmpeg to capture a portion of a yt video
# $1 - youtube URL
# $2 - start position in hh:mm:ss.msms format (ms=miliseconds)
# $3 - final position in hh:mm:ss.msms format (ms=miliseconds)
# $4 - output file name (optional)
# $5 - output video codec type (optional, for instance: libx264)
# $6 - output audio codec type (optional, for instance: aac)
function youtubedl_snippet()(
  local url_streams=$(youtube-dl -f best --get-url $1)
  local output_name=$(youtube-dl --get-title $1)

  ffmpeg -ss $2 -to $3 -i $url_streams -c:v copy -c:a copy ${4:-"$output_name.mp4"}
)

function stopwatch()
{
  local start=$SECONDS
  command $@
  echo "it takes $(( SECONDS - start )) seconds"
}
