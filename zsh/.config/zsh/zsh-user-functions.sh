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


# function to perform sparse clone, to see more, check this out
# https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository
function git_sparse_clone() (
  rurl="$1" localdir="$2" && shift 2

  mkdir -p "$localdir"
  cd "$localdir"

  git init
  git remote add -f origin "$rurl"

  git config core.sparseCheckout true

  # Loops over remaining args
  for i; do
    echo "$i" >> .git/info/sparse-checkout
  done

  git pull origin master
)


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

  # url_array=("${(@f)$(echo $url_streams)}") # split the urls by lines using function
  local url_array=(${(f)url_streams}) # split the urls by lines url_array[1] -> video stream url_array[2] -> audio stream

  # TODO: understand why it didn't worked out
  # ffmpeg -ss $2 -to $3 -i ${url_array[1]} -ss $2 -to $3 -i ${url_array[2]} -map 0:v -map 1:a -c:v ${5:-copy} -c:a ${6:-copy} ${4:-"$output_name.mp4"}
  ffmpeg -ss $2 -to $3 -i $url_streams -c:v copy -c:a copy ${4:-"$output_name.mp4"}
)
