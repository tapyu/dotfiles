#!/bin/sh

# here you put the shell(local) variables
# History in cache directory:
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=${XDG_STATE_HOME:-$HOME/.local/state}/zsh/history

# less config
# it must be here since the variable $TERM shall be defined first
export LESS=-R
export LESS_TERMCAP_mb="$(printf '\33[1;31m')"
export LESS_TERMCAP_md="$(printf '\33[1;36m')"
export LESS_TERMCAP_me="$(printf '\33[0m')"
export LESS_TERMCAP_so="$(printf '\33[01;44;33m')"
export LESS_TERMCAP_se="$(printf '\33[0m')"
export LESS_TERMCAP_us="$(printf '\33[1;32m')"
export LESS_TERMCAP_ue="$(printf '\33[0m')"
export LESSOPEN="| /usr/bin/highlight -O ansi %s 2>/dev/null"
