# enable color support of ls and also add handy aliases
if [[ -x /usr/bin/dircolors ]]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Verbosity and settings that you pretty much just always are going to want.
# TODO: see more here: https://github.com/LukeSmithxyz/voidrice/blob/master/.config/shell/aliasrc
alias cp='cp --interactive --verbose'
alias mv='mv --interactive --verbose'
alias mkdir='mkdir --parents --verbose'

# sharkdp/bat
alias cat=bat
# dandavison/delta
alias diff=delta
# wezterm image redering capability
alias imgcat='wezterm imgcat'
# peltoche/lsd program
[[ "$(command -v lsd)" ]] && alias ls='lsd --group-dirs first --total-size' # although `--total-size` is useless without the `-l optionl`, it must be placed here as `\ls` has no `--total-size` option
alias l='ls'
alias l1='ls -1'
alias ll='ls -l --almost-all --human-readable' # -l -> long list formating; --almost-all -> ignore . and ..

# nnn
alias nnn='n_cd -e -P p'
alias n='nnn'

alias rm='safe-rm'

# fzf-based aliases
alias fop='command ${EDITOR:-vi} "$(fzf)"' # open file through fzf command
