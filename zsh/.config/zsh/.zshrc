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
bindkey '^E' edit-command-line

# open nnn with ^o
bindkey -s '^o' 'n\n'

# all base functions of zsh, this makes all the job of the "oh-my-zsh" bloated stuff
source "$ZDOTDIR/zsh-base-functions.sh"

# user files to source
source "$ZDOTDIR/zsh-exports.sh"
source "$ZDOTDIR/zsh-aliases.sh"
source "$ZDOTDIR/zsh-user-functions.sh"

# plugins
zsh_add_plugin "zsh-users/zsh-syntax-highlighting"
zsh_add_plugin "hlissner/zsh-autopair"
zsh_add_plugin "zsh-users/zsh-autosuggestions"
zsh_add_plugin "MenkeTechnologies/zsh-expand"
zsh_add_plugin "zsh-users/zsh-history-substring-search"
zsh_add_plugin "joshskidmore/zsh-fzf-history-search"
mkdir -p $ZDOTDIR/plugins/git-auto-fetch; [[ ! -f $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh ]] && curl https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh > $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh # git-auto-fetch oh-my-zsh plugin
zsh_add_plugin "laggardkernel/git-ignore"
# zsh_add_plugin "egyptianbman/zsh-git-worktrees" # bloated plugin!
# zsh_add_plugin "MichaelAquilina/zsh-you-should-use" # not necessary anymore (?)

# theme
zsh_add_theme "romkatv/powerlevel10k" # to customize run `p10k configure` or edit ~/.config/zsh/.p10k.zsh.
source $ZDOTDIR/.p10k.zsh

# zsh-history-substring-search plugin setup
bindkey "^[[A" history-substring-search-up   # ^[[A -> up arrow
bindkey "^[[B" history-substring-search-down # ^[[B -> down arrow

# zoxide setup
eval "$(zoxide init zsh)"

## execute tmux as soon as the shell opens up
#if command -v tmux &> /dev/null && [ -n "$PS1" ] && [[ ! "$TERM" =~ screen ]] && [[ ! "$TERM" =~ tmux ]] && [ -z "$TMUX" ]; then
#  exec tmux
#fi

# # pyenv settings
# if [[ $PATH =~ pyenv ]]; then
#   eval "$(pyenv init --path)"
#   eval "$(pyenv virtualenv-init -)"
# fi