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
  local branch=$(git branch --show-current)
  
  git add -A
  git commit -m $1
  git push origin $branch
}
