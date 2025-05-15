### basic config ###
setopt appendhistory         # append new history entries to the history file, rather than overwriting the entire file each time the shell exits
setopt autocd                # if you type a directory name as a command and press Enter, Zsh automatically changes to that directory
setopt extendedglob          # this option extends the functionality of globbing patterns in zsh, allowing the use of additional features such	^(foo|bar)
setopt nomatch               # prevents the zsh from reporting an error if a glob pattern does not match any files
setopt interactive_comments  # allows you to include comments in interactive shell sessions
unsetopt BEEP                # unset beep
setopt HIST_IGNORE_ALL_DUPS  # prevent duplicate entries in history
setopt HIST_APPEND           # append history instead of overwriting
setopt INC_APPEND_HISTORY    # save each command as it is executed
setopt SHARE_HISTORY         # share history across multiple sessions
stty stop undef	             # disable ctrl-s to freeze terminal.
zle_highlight=('paste:none') # disable highlighting when pasting something

### auto/tab complete ###
autoload -U compinit # load functions or scripts on demand. compinit is a function that initializes the zsh completion system
compinit
zmodload zsh/complist # load modules (AKA shared object files) that extend the functionality of the shell. complist provides additional capabilities related to tab-completion and completion listing
# enable to tab-complete dotfiles
_comp_options+=(globdots) # Include hidden files.

### keybinds ###
# edit line in $EDITOR with ctrl-e:
autoload edit-command-line
zle -N edit-command-line
bindkey '^E' edit-command-line
# open nnn with ^o
bindkey -s '^o' 'n\n'

### imports ###
source "$ZDOTDIR/base_functions.sh" # this makes all the jobs of the bloated "oh-my-zsh" stuff
source "$ZDOTDIR/exports.sh"
source "$ZDOTDIR/aliases.sh"
source "$ZDOTDIR/user_functions.sh"

### plugins ###
zsh_add_plugin "zsh-users/zsh-syntax-highlighting"
zsh_add_plugin "hlissner/zsh-autopair"
zsh_add_plugin "zsh-users/zsh-autosuggestions"
zsh_add_plugin "MenkeTechnologies/zsh-expand" # another possible alternative to it is `olets/zsh-abbr`, which has more stars
zsh_add_plugin "zsh-users/zsh-history-substring-search"
zsh_add_plugin "joshskidmore/zsh-fzf-history-search"
mkdir -p $ZDOTDIR/plugins/git-auto-fetch; [[ ! -f $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh ]] && curl https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh > $ZDOTDIR/plugins/git-auto-fetch/git-auto-fetch.plugin.zsh # git-auto-fetch oh-my-zsh plugin
zsh_add_plugin "laggardkernel/git-ignore"
zsh_add_plugin "Aloxaf/fzf-tab"
zsh_add_plugin "matthiasha/zsh-uv-env"

### plugins setup ###
# zsh-history-substring-search
bindkey "^[[1;5A" history-substring-search-up   # ^[[A -> up arrow
bindkey "^[[1;5B" history-substring-search-down # ^[[B -> down arrow

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

# cdhist SEE: https://github.com/bulletmark/cdhist
if type cdhist &>/dev/null; then
    source <(cdhist -i)
fi
source <(fzf --zsh | sed 's/builtin cd/cd/g') # NOTE: changes the `alt+C` (a `zsh` shortcut) from `builtin cd` to `cdhist`. In practice, it makes `alt+C` run `cd --`
