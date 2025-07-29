# add plugin, it basically does what the bloated oh-my-zsh does
zsh_add_plugin() {
  PLUGIN_NAME=$(echo $1 | cut -d "/" -f 2 | cut -d '.' -f 1)
  [[ ! -d "$ZDOTDIR/plugins/$PLUGIN_NAME" ]] && git clone "https://github.com/$1" "$ZDOTDIR/plugins/$PLUGIN_NAME"
  for script in \
    "$ZDOTDIR/plugins/$PLUGIN_NAME/$PLUGIN_NAME.plugin.zsh" \
    "$ZDOTDIR/plugins/$PLUGIN_NAME/$PLUGIN_NAME.zsh" \
    "$ZDOTDIR/plugins/$PLUGIN_NAME/${PLUGIN_NAME#zsh-}.plugin.zsh"
  do
    [[ -r $script ]] && source $script && break
  done
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
