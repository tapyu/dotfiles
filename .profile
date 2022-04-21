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
    PATH="$HOME/.local/bin:$PATH"
fi

# User specific aliases and functions
# To add directories to your PATH or define additional environment variables, place those changes in .bash_profile (or the equivalent, according to your distribution; for example, Ubuntu uses .profile). For everything else, place the changes in .bashrc.

# XDG Paths
export XDG_CONFIG_HOME="$HOME/.config" # Where user-specific configurations should be written (analogous to /etc)
export XDG_CACHE_HOME="$HOME/.cache" # Where user-specific non-essential (cached) data should be written (analogous to /var/cache)
export XDG_DATA_HOME="$HOME/.local/share" # Where user-specific data files should be written (analogous to /usr/share)
export XDG_STATE_HOME="$HOME/.local/state" # Where user-specific state files should be written (analogous to /var/lib)

## tidy up $HOME directory
export NPM_CONFIG_USERCONFIG="$XDG_DATA_HOME/npm" # change ~/.npm/ directory (?)
export ZDOTDIR="$XDG_CONFIG_HOME/zsh" # set zsh default config directory
export __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nv" # change proprietary nvidia drivers (~/.nv/) directory (?)
export GIT_CONFIG="$XDG_CONFIG_HOME/gitconfig" # change .gitconfig (?)
export LESSHISTFILE="$XDG_STATE_HOME/less/history" # change ~/.lesshst
export CARGO_HOME="$XDG_DATA_HOME/cargo" # change ~/.cargo (?)
export JULIA_DEPOT_PATH="$XDG_DATA_HOME/julia" # change ~/.julia (?)
[ -d $CARGO_HOME ] && [ source $CARGO_HOME ] # source cargo, the Rust package manager

## default programs
export EDITOR="nvim" # set editor
export BROWSER="brave" # set browser
export MANPAGER='nvim +Man!' # set manpager

## program settings
export MATLAB_ROOT_DIR="/usr/local/Polyspace/R2021a" # matlab root directory
# zoxide environment variables
export _ZO_DATA_DIR="$XDG_DATA_HOME/zoxide"
export _ZO_RESOLVE_SYMLINKS='1'
export RANGER_DEVICONS_SEPARATOR="  " # ranger conf - icon and directory name separator
# set zsh-users/zsh-history-substring-search plugin
export HISTORY_SUBSTRING_SEARCH_HIGHLIGHT_FOUND=''
export HISTORY_SUBSTRING_SEARCH_PREFIXED='true'
export FZF_DEFAULT_OPTS="--layout=reverse --height 40%"

# Julia's env variable - where the package manager look for package registries, installed packages, named environments, repo clones, cached compiled package images, configuration files, and the default location of the REPL's history file.
export JULIA_HISTORY="$XDG_CACHE_HOME/julia"
