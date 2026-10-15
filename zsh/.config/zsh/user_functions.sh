cpmkdir() {
  # usage: cpmkdir [cp options...] SRC... DEST
  # Creates DEST dir (or DEST's parent, if DEST is a file path), then runs cp.

  local dest
  dest="${@: -1}"          # last argument = DEST

  # If DEST ends with / or is an existing directory, ensure it exists.
  if [[ "$dest" == */ || -d "$dest" ]]; then
    mkdir -p -- "$dest" || return 1
  else
    # DEST is a file path (or non-existent dir without trailing /): create its parent
    mkdir -p -- "$(dirname -- "$dest")" || return 1
  fi

  command cp -aiv -- "$@" # `command` avoids recursion
}


git-sparse-clone() {
  # function to perform sparse clone, to see more, check this out
  # https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository
  # Arguments:
  #   $1      -> repository url
  #   $2      -> repository branch
  #   $3      -> local directory path (include dir to be created)
  #   $4...   -> subdirectories cloned from the repo url
  #######################################
  repo_url="$1" branch="$2" localdir="$3" && shift 3

  mkdir -p "$localdir"
  cd "$localdir"

  git init
  git config core.sparseCheckout true
  git remote add -f origin "$repo_url"

  # Loops over remaining args
  for i; do
    echo "$i" >> .git/info/sparse-checkout
  done

  git pull origin $branch
}

gwt() {
  # cd between the git worktrees
  local worktree
  worktree=$(git worktree list | fzf | awk '{print $1}')
  if [ -n "$worktree" ]; then
      cd "$worktree" || echo "Failed to navigate to $worktree"
  else
      echo "No worktree selected."
  fi
}

fp() {
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

ft() {
  # ft [FUZZY PATTERN] - Open the selected text file with the default editor
  #   - Bypass fuzzy finder if there's only one match (--select-1)
  #   - Exit if there's no match (--exit-0)
  IFS=$'\n' files=($(fzf-tmux --query="$1" --multi --select-1 --exit-0))
  [[ -n "$files" ]] && ${EDITOR:-hx:-vim:-vi} "${files[@]}"
}

function pet-select() {
  # knqyf263/pet function
  # NOTE: It is binded to ^S
  # SEE: zshrc.sh
  BUFFER=$(pet search --query "$LBUFFER")
  CURSOR=$#BUFFER
  zle redisplay
}
