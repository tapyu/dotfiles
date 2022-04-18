# some useful options (man zshoptions)
setopt appendhistory
setopt autocd extendedglob nomatch menucomplete
setopt interactive_comments
unsetopt BEEP # unset beep
stty stop undef	# Disable ctrl-s to freeze terminal.
zle_highlight=('paste:none') # disable highlighting when pasting something

# basic auto/tab complete:
autoload -U compinit
zstyle ':completion:*' menu select
zmodload zsh/complist
compinit
# enable to tab-complete dotfiles
_comp_options+=(globdots) # Include hidden files.


# vi mode
bindkey -v
export KEYTIMEOUT=1
# use vim keys in tab complete menu:
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history
bindkey -v '^?' backward-delete-char
# Edit line in vim with ctrl-e:
autoload edit-command-line
zle -N edit-command-line
bindkey '^ ' edit-command-line


# all base functions of zsh, this makes all the job of the "oh-my-zsh" bloated stuff
source "$ZDOTDIR/zsh-base-functions.sh"

# user files to source
zsh_add_file "zsh-exports.sh"
zsh_add_file "zsh-aliases.sh"
zsh_add_file "zsh-user-functions.sh"

# plugins
zsh_add_plugin "zsh-users/zsh-autosuggestions"
zsh_add_plugin "zsh-users/zsh-syntax-highlighting"
zsh_add_plugin "hlissner/zsh-autopair"
zsh_add_plugin "romkatv/powerlevel10k"
zsh_add_plugin "zsh-users/zsh-history-substring-search"
zsh_add_plugin "joshskidmore/zsh-fzf-history-search"


# apply p10k
source $ZDOTDIR/plugins/powerlevel10k/powerlevel10k.zsh-theme # apply powerlevel10ktheme
# to customize prompt, run `p10k configure` or edit ~/.config/zsh/.p10k.zsh.
[[ ! -f ~/.config/zsh/.p10k.zsh ]] || source ~/.config/zsh/.p10k.zsh


# zsh-history-substring-search plugin setup
bindkey "^[[A" history-substring-search-up
bindkey "^[[B" history-substring-search-down

# zoxide setup
eval "$(zoxide init zsh)"
