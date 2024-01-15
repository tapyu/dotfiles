# add plugin, it basically does what the bloated oh-my-zsh does
zsh_add_plugin() {
  PLUGIN_NAME=$(echo $1 | cut -d "/" -f 2 | cut -d '.' -f 1)
  [[ ! -d "$ZDOTDIR/plugins/$PLUGIN_NAME" ]] && git clone "https://github.com/$1" "$ZDOTDIR/plugins/$PLUGIN_NAME"
  source "$ZDOTDIR/plugins/$PLUGIN_NAME/$PLUGIN_NAME.plugin.zsh" 2> /dev/null || source "$ZDOTDIR/plugins/$PLUGIN_NAME/$PLUGIN_NAME.zsh"
}

# add theme
zsh_add_theme() {
  THEME_NAME=$(echo $1 | cut -d "/" -f 2)
  [[ ! -d "$ZDOTDIR/theme" ]] && git clone "https://github.com/$1" "$ZDOTDIR/theme/$THEME_NAME"
  source "$ZDOTDIR/theme/$THEME_NAME/$THEME_NAME.zsh-theme"
}

# function to add completions to zsh, see these completions here
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