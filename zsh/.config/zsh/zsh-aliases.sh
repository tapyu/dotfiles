# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    # Peltoche/lsd program
    alias ls='lsd --group-dirs first'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# matlab startup program
#alias matlab='$MATLAB_ROOT_DIR/bin/matlab >> /dev/null 2>> /dev/null &'
# sharkdp/bat program
alias cat=bat
# dandavison/delta program
alias diff=delta
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'
alias v=nvim

# Verbosity and settings that you pretty much just always are going to want.
# TODO see more here
# https://github.com/LukeSmithxyz/voidrice/blob/master/.config/shell/aliasrc
# -l -> long list formating
# --almost-all -> ignore . and ..
alias \
	cp="cp --interactive --verbose" \
	mv="mv --interactive --verbose" \
	alias ll='ls -l --almost-all --human-readable' \ 
	alias la='ls --almost-all' \ 
	alias l='ls --human-readable' \
	alias mkdir="mkdir --parents --verbose"
