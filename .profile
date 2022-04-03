# ~/.profile: executed by the command interpreter for login shells.
# all Enviromental (global) variables must be set here
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# the default umask is set in /etc/profile; for setting the umask
# for ssh logins, install and configure the libpam-umask package.
#umask 022
# echo "from ~/.profile" # just testing this file

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin/" ] ; then
    PATH="$HOME/.local/bin/:$PATH"
fi

# User specific aliases and functions
# To add directories to your PATH or define additional environment variables, place those changes in .bash_profile (or the equivalent, according to your distribution; for example, Ubuntu uses .profile). For everything else, place the changes in .bashrc.

# matlab root directorie
export MATLAB_ROOT_DIR="/usr/local/Polyspace/R2021a"
# ranger conf - icon and directory name separator
export RANGER_DEVICONS_SEPARATOR="  "
# source cargo, the Rust package manager
source "$HOME/.cargo/env"
# set zsh default condig directory
export ZDOTDIR=$HOME/.config/zsh # without / to avoid error
# XDG Paths
export XDG_CONFIG_HOME=$HOME/.config
export XDG_CACHE_HOME=$HOME/.cache
export XDG_DATA_HOME=$HOME/.local/share
# set editor
export EDITOR="nvim"
# set browser
export BROWSER="brave"
# set manpager
export MANPAGER='nvim +Man!'

# set zsh-history-substring-search plugin
export HISTORY_SUBSTRING_SEARCH_HIGHLIGHT_FOUND=''
export HISTORY_SUBSTRING_SEARCH_PREFIXED='true'
export FZF_DEFAULT_OPTS="--layout=reverse --height 40%"
