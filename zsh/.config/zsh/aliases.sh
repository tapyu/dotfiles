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
alias cp='cp --interactive --recursive'
alias mv='mv --interactive --verbose'
alias mkdir='mkdir --parents --verbose'
alias l='ls'
alias l1='ls --oneline'
alias l1a='ls --almost-all --oneline'
alias la='ls --almost-all'
alias ll='ls -l --almost-all --human-readable' # -l -> long list formating; --almost-all -> ignore . and ..

# BurntSushi/ripgrep
alias grep='rg' # ripgrep
# sharkdp/bat
alias cat='bat'
# dandavison/delta
alias diff='delta'
# wezterm image redering capability
alias imgcat='wezterm imgcat'
# peltoche/lsd
[[ "$(command -v lsd)" ]] && alias ls='lsd --group-dirs first --total-size' # although `--total-size` is useless without the `-l optionl`, it must be placed here as `\ls` has no `--total-size` option
# jarun/nnn
alias nnn='n_cd -e -P p'
alias n='nnn'
# kaelzhang/shell-safe-rm
alias rm='safe-rm -rf'
# sharkdp/fd
alias fd='fdfind'

