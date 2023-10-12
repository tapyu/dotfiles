#######################################
# function to perform sparse clone, to see more, check this out
# https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository
# Arguments:
#   $1      -> repository url
#   $2      -> local directory path (include dir to be created)
#   $3...   -> subdirectories cloned from the repo url
#######################################
git_sparse_clone() {
  repo_url="$1" localdir="$2" && shift 2

  mkdir -p "$localdir"
  cd "$localdir"

  git init
  git config core.sparseCheckout true
  git remote add -f origin "$repo_url"

  # Loops over remaining args
  for i; do
    echo "$i" >> .git/info/sparse-checkout
  done

  git pull origin main
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
  dir=$(find ${1:-.} -path '*/\.*' -prune \
                  -o -type d -print 2> /dev/null | fzf +m) &&
  cd "$dir"
}