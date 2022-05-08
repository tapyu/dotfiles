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
# $4 - output file name (optitional)
function youtubedl_snippet()(
  local url_streams=$(youtube-dl --get-url $1)
  local output_name=$(youtube-dl --get-title $1)

  # url_array=("${(@f)$(echo $url_streams)}") # split the urls by lines using function
  url_array=(${(f)url_streams}) # split the urls by lines url_array[1] -> video stream url_array[2] -> audio stream

  ffmpeg -ss $2 -to $3 -i ${url_array[1]} -ss $2 -to $3 -i ${url_array[2]} -map 0:v -map 1:a -c:v libx264 -c:a aac ${4:-output.mkv}
)
