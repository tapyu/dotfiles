#######################################
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

### fzf-based commads
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

# fd - cd to selected directory
fd() {
  dir=$(find ${1:-.} -path '*/\.*' -prune \
                  -o -type d -print 2> /dev/null | fzf +m) &&
  cd "$dir"
}

n_cd () {
    # Block nesting of nnn in subshells
    [ "${NNNLVL:-0}" -eq 0 ] || {
        echo "nnn is already running"
        return
    }

    # The behaviour is set to cd on quit (nnn checks if NNN_TMPFILE is set)
    # If NNN_TMPFILE is set to a custom path, it must be exported for nnn to
    # see. To cd on quit only on ^G, remove the "export" and make sure not to
    # use a custom path, i.e. set NNN_TMPFILE *exactly* as follows:
    #      NNN_TMPFILE="${XDG_CONFIG_HOME:-$HOME/.config}/nnn/.lastd"
    export NNN_TMPFILE="${XDG_CONFIG_HOME:-$HOME/.config}/nnn/.lastd"

    # Unmask ^Q (, ^V etc.) (if required, see `stty -a`) to Quit nnn
    # stty start undef
    # stty stop undef
    # stty lwrap undef
    # stty lnext undef

    # The command builtin allows one to alias nnn to n, if desired, without
    # making an infinitely recursive alias
    command nnn "$@"

    [ ! -f "$NNN_TMPFILE" ] || {
        . "$NNN_TMPFILE"
        rm -f "$NNN_TMPFILE" > /dev/null
    }
}
