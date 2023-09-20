# enable color support of ls and also add handy aliases
if [[ -x /usr/bin/dircolors ]]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# sharkdp/bat
alias cat=bat
# dandavison/delta
alias diff=delta

# Verbosity and settings that you pretty much just always are going to want.
# TODO: see more here: https://github.com/LukeSmithxyz/voidrice/blob/master/.config/shell/aliasrc
alias cp='cp --interactive --verbose'
alias mv='mv --interactive --verbose'
alias mkdir='mkdir --parents --verbose'
# peltoche/lsd program
[[ -x $(which lsd) ]] && alias ls='lsd --group-dirs first --total-size' # although `--total-size` is useless without the `-l optionl`, it must be placed here as `\ls` has no `--total-size` option
alias l='ls'
alias l1='ls -1'
alias ll='ls -l --almost-all --human-readable' # -l -> long list formating; --almost-all -> ignore . and ..

# fzf-based aliases
alias fop='xdg-open "$(fzf)"' # open file through fzf command
