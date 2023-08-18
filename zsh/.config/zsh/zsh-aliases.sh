# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    # Peltoche/lsd program
    alias ls='lsd --group-dirs first --total-size'
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
# TODO see more here: https://github.com/LukeSmithxyz/voidrice/blob/master/.config/shell/aliasrc
alias cp="cp --interactive --verbose"
alias mv="mv --interactive --verbose"
alias ll='ls -l --almost-all --human-readable' # -l -> long list formating; --almost-all -> ignore . and ..
alias la='ls --almost-all'
alias l='ls --human-readable'
alias mkdir="mkdir --parents --verbose"

# fzf-based aliases
alias fop='xdg-open "$(fzf)"' # open file through fzf command
