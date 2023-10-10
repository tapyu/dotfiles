#######################################
# function to perform sparse clone, to see more, check this out
# https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository
# Arguments:
#   $1      -> repository url
#   $2      -> local directory path (include dir to be created)
#   $3...   -> subdirectories cloned from the repo url
#######################################
git_sparse_clone() (
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
)

# Function to source files if they exist
zsh_add_file() {
    [ -f "$ZDOTDIR/$1" ] && source "$ZDOTDIR/$1"
}

# add a plugin, it is basically what oh-my-zsh does
zsh_add_plugin() {
  if [ $# -gt 1 ]; then # if the number of arguments is greater than 1, make sparse clone; else, make clone
    # git sparse clone
    PLUGIN_NAME=$(echo $3 | cut -d "/" -f 2 | cut -d '.' -f 1)
    if [ -d "$ZDOTDIR/plugins/$PLUGIN_NAME" ]; then # if the directory already exists, add pluign
      zsh_add_file "plugins/$PLUGIN_NAME/$PLUGIN_NAME.plugin.zsh" || \
      zsh_add_file "plugins/$PLUGIN_NAME/$PLUGIN_NAME.zsh"
    else
      git_sparse_clone $@
      mkdir "$ZDOTDIR/plugins/$PLUGIN_NAME" 
      cp $2/$3/* "$ZDOTDIR/plugins/$PLUGIN_NAME"
    fi
  else
    # git clone
    PLUGIN_NAME=$(echo $1 | cut -d "/" -f 2 | cut -d '.' -f 1)
    if [ -d "$ZDOTDIR/plugins/$PLUGIN_NAME" ]; then # if the directory already exists, add plugin
      zsh_add_file "plugins/$PLUGIN_NAME/$PLUGIN_NAME.plugin.zsh" || \
      zsh_add_file "plugins/$PLUGIN_NAME/$PLUGIN_NAME.zsh"
    else
      git clone "https://github.com/$1.git" "$ZDOTDIR/plugins/$PLUGIN_NAME"
    fi
  fi
}

# function to add extension to zsh, see these completions here
# https://github.com/unixorn/awesome-zsh-plugins#completions
zsh_add_completion() {
    PLUGIN_NAME=$(echo $1 | cut -d "/" -f 2 | cut -d '.' -f 1)
    if [ -d "$ZDOTDIR/plugins/$PLUGIN_NAME" ]; then 
        # For completions
		completion_file_path=$(ls $ZDOTDIR/plugins/$PLUGIN_NAME/_*)
		fpath+="$(dirname "${completion_file_path}")"
        zsh_add_file "plugins/$PLUGIN_NAME/$PLUGIN_NAME.plugin.zsh"
    else
        git clone "https://github.com/$1.git" "$ZDOTDIR/plugins/$PLUGIN_NAME"
		fpath+=$(ls $ZDOTDIR/plugins/$PLUGIN_NAME/_*)
        [ -f $ZDOTDIR/.zccompdump ] && $ZDOTDIR/.zccompdump
    fi
	completion_file="$(basename "${completion_file_path}")"
	if [ "$2" = true ] && compinit "${completion_file:1}"
}


# Change cursor shape for different vi modes.
# Executed every time the keymap changes, i.e. the special parameter KEYMAP
# is set to a different value
zle-keymap-select() {
  if [[ ${KEYMAP} == vicmd ]] ||
     [[ $1 = 'block' ]]; then
    echo -ne '\e[1 q' # block shape
  elif [[ ${KEYMAP} == main ]] ||
       [[ ${KEYMAP} == viins ]] ||
       [[ ${KEYMAP} = '' ]] ||
       [[ $1 = 'beam' ]]; then
    echo -ne '\e[6 q' # beam shape
  fi
}

zle-line-init() {
  zle -K viins # initiate `vi insert` as keymap (can be removed if `bindkey -V` has been set elsewhere)
  # echo -ne "\e[6 q" # beam shape
}
zle -N zle-line-init
zle -N zle-keymap-select
# echo -ne '\e[6 q' # Use beam shape cursor on startup. (use \e[5 for blinking beam shape
# preexec() { echo -ne '\e[6 q' ;} # Use beam shape cursor for each new prompt.