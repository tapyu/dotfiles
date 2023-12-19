### basic config ###
setopt appendhistory
setopt autocd extendedglob nomatch menucomplete
setopt interactive_comments
unsetopt BEEP # unset beep
stty stop undef	# Disable ctrl-s to freeze terminal.
zle_highlight=('paste:none') # disable highlighting when pasting something

### auto/tab complete ###
autoload -U compinit
zstyle ':completion:*' menu select
zmodload zsh/complist
compinit
# enable to tab-complete dotfiles
_comp_options+=(globdots) # Include hidden files.

### vi mode ###
bindkey -v
export KEYTIMEOUT=1
# use vim keys in tab complete menu:
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history
bindkey -v '^?' backward-delete-char # `del` key (denoted by ^?) to delete a char in vi mode

### keybinds ###
# edit line in $EDITOR with ctrl-e:
autoload edit-command-line
zle -N edit-command-line
bindkey '^E' edit-command-line
# open nnn with ^o
bindkey -s '^o' 'n\n'

### imports ###
source "$ZDOTDIR/zsh-base-functions.sh" # this makes all the job of the "oh-my-zsh" bloated stuff
source "$ZDOTDIR/zsh-exports.sh"
source "$ZDOTDIR/zsh-aliases.sh"
source "$ZDOTDIR/zsh-user-functions.sh"

### plugins ###
zsh_add_plugin "zsh-users/zsh-syntax-highlighting"
zsh_add_plugin "hlissner/zsh-autopair"
zsh_add_plugin "zsh-users/zsh-autosuggestions"
zsh_add_plugin "MenkeTechnologies/zsh-expand"
zsh_add_plugin "zsh-users/zsh-history-substring-search"
zsh_add_plugin "joshskidmore/zsh-fzf-history-search"
mkdir -p $ZDOTDIR/plugins/git-auto-fetch; [[ ! -f $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh ]] && curl https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh > $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh # git-auto-fetch oh-my-zsh plugin
zsh_add_plugin "laggardkernel/git-ignore"
zsh_add_plugin "tapyu/zsh-activate-py-environment"
# zsh_add_plugin "egyptianbman/zsh-git-worktrees" # bloated plugin!
# zsh_add_plugin "MichaelAquilina/zsh-you-should-use" # not necessary anymore (?)

### plugins setup ###
# zsh-history-substring-search
bindkey "^[[A" history-substring-search-up   # ^[[A -> up arrow
bindkey "^[[B" history-substring-search-down # ^[[B -> down arrow

### theme ###
zsh_add_theme "romkatv/powerlevel10k" # to customize run `p10k configure` or edit ~/.config/zsh/.p10k.zsh.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi
[[ ! -f ~/.config/zsh/.p10k.zsh ]] || source ~/.config/zsh/.p10k.zsh

### software setups ###
# zoxide
eval "$(zoxide init zsh)"
# tmux
#if command -v tmux &> /dev/null && [ -n "$PS1" ] && [[ ! "$TERM" =~ screen ]] && [[ ! "$TERM" =~ tmux ]] && [ -z "$TMUX" ]; then
#  exec tmux
#fi
# navi
eval "$(navi widget zsh)"
