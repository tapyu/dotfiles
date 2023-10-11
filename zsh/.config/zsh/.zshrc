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
# edit line in $EDITOR with ctrl space:
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
zsh_add_plugin "https://github.com/ohmyzsh/ohmyzsh" /tmp/git-auto-fetch plugins/git-auto-fetch # sparse checkout
zsh_add_plugin "laggardkernel/git-ignore"
zsh_add_plugin "romkatv/powerlevel10k"
zsh_add_plugin "hlissner/zsh-autopair"
zsh_add_plugin "zsh-users/zsh-autosuggestions"
zsh_add_plugin "joshskidmore/zsh-fzf-history-search"
# zsh_add_plugin "MenkeTechnologies/zsh-expand"
zsh_add_plugin "egyptianbman/zsh-git-worktrees"
zsh_add_plugin "MichaelAquilina/zsh-you-should-use"
zsh_add_plugin "zsh-users/zsh-history-substring-search"
zsh_add_plugin "zsh-users/zsh-syntax-highlighting"

# apply p10k
source $ZDOTDIR/plugins/powerlevel10k/powerlevel10k.zsh-theme # apply powerlevel10ktheme
# to customize prompt, run `p10k configure` or edit ~/.config/zsh/.p10k.zsh.
[[ ! -f ~/.config/zsh/.p10k.zsh ]] || source ~/.config/zsh/.p10k.zsh

# zsh-history-substring-search plugin setup
bindkey "^[[A" history-substring-search-up   # ^[[A -> up arrow
bindkey "^[[B" history-substring-search-down # ^[[B -> down arrow

# zoxide setup
eval "$(zoxide init zsh)"

## execute tmux as soon as the shell opens up
#if command -v tmux &> /dev/null && [ -n "$PS1" ] && [[ ! "$TERM" =~ screen ]] && [[ ! "$TERM" =~ tmux ]] && [ -z "$TMUX" ]; then
#  exec tmux
#fi

# pyenv settings
if [[ $PATH =~ pyenv ]]; then
  eval "$(pyenv init --path)"
  eval "$(pyenv virtualenv-init -)"
fi