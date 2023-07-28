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
youtubedl_snippet()(
  local url_streams=$(youtube-dl -f best --get-url $1)
  local output_name=$(youtube-dl --get-title $1)

  ffmpeg -ss $2 -to $3 -i $url_streams -c:v copy -c:a copy ${4:-"$output_name.mp4"}
)

# count how many secs it takes to run a command
stopwatch()
{
  local start=$SECONDS
  command $@
  echo "it takes $(( SECONDS - start )) seconds"
}


### fzf-based commads
fp()
{
  # open PDF using fuzzy finder
  local open=xdg-open   # this will open pdf file withthe default PDF viewer on KDE, xfce, LXDE and perhaps on other desktops.
  rg --type 'pdf' --files \
  | fast-p \
  | fzf --read0 --reverse -e -d $'\t'  \
      --preview-window down:80% --preview '
          v=$(echo {q} | tr " " "|"); 
          echo -e {1}"\n"{2} | grep -E "^|$v" -i --color=always;
      ' \
  | cut -z -f 1 -d $'\t' | tr -d '\n' | xargs -r --null $open > /dev/null 2> /dev/null
}

# ft [FUZZY PATTERN] - Open the selected text file with the default editor
#   - Bypass fuzzy finder if there's only one match (--select-1)
#   - Exit if there's no match (--exit-0)
ft() {
  IFS=$'\n' files=($(fzf-tmux --query="$1" --multi --select-1 --exit-0))
  [[ -n "$files" ]] && ${EDITOR:-hx:-vim:-vi} "${files[@]}"
}

# fd - cd to selected directory
fd() {
 cal dir
  dir=$(find ${1:-.} -path '*/\.*' -prune \
                  -o -type d -print 2> /dev/null | fzf +m) &&
  cd "$dir"
}

# git add, commit and push to the current branch
# $1 -> git commit message
gp() {
  local branch=$(git branch | sed -n '1s/^\*\s//p')
  
  git add -A
  git commit -m $1
  git push origin $branch
}
