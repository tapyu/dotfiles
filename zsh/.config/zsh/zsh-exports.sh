#!/bin/sh

# here you put the shell(local) variables
# History in cache directory:
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=${XDG_STATE_HOME:-$HOME/.local/state}/zsh/history
