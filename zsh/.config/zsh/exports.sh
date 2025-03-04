#!/bin/sh

## here you put the shell(local) variables
## the settings here are technically shell variables, not exported environment variables.
# History in cache directory:
HISTSIZE=150000
SAVEHIST=150000
HISTFILE=${XDG_STATE_HOME:-$HOME/.local/state}/zsh/history

# groff (GNU roff) is a typesetting system used to format and display `man` pages.
# By default, groff uses SGR (Select Graphic Rendition), an ANSI standard that controls text formatting,
# such as bold, underlining, and color attributes. SGR is useful for modern terminal emulators
# but can cause issues when used with `less`, as it may not interpret SGR sequences correctly.
#
# Setting GROFF_NO_SGR=1 tells groff to avoid using SGR sequences and instead use traditional
# ANSI escape codes. This ensures that `man` pages display properly with color formatting
# when viewed through `less -R`, making bold and underlined text visible as expected.
export GROFF_NO_SGR=1

# NOTE: as an alternative, If colors are still not working, try disabling
# SGR sequences. This forces groff to use older escape sequences instead
# of SGR formatting. Uncomment the line below if GROFF_NO_SGR does not work
# export MANROFFOPT="-c"

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

