# ~/.profile: executed by the command interpreter for login shells.
# all Enviromental (global) variables must be set here
# This file is not read by bash(1), if ~/.bash_profile or ~/.bash_login
# exists.
# see /usr/share/doc/bash/examples/startup-files for examples.
# the files are located in the bash-doc package.

# the default umask is set in /etc/profile; for setting the umask
# for ssh logins, install and configure the libpam-umask package.
#umask 022

# XDG paths
export XDG_CONFIG_HOME="$HOME/.config" # Where user-specific configurations should be written (analogous to /etc)
export XDG_CACHE_HOME="$HOME/.cache" # Where user-specific non-essential (cached) data should be written (analogous to /var/cache)
export XDG_DATA_HOME="$HOME/.local/share" # Where user-specific data files should be written (analogous to /usr/share)
export XDG_STATE_HOME="$HOME/.local/state" # Where user-specific state files should be written (analogous to /var/lib)

# add to $PATH bin directories
[[ -d "$HOME/.local/bin" ]] && [[ ":$PATH:" != *"$HOME/.local/bin"* ]] && PATH="$HOME/.local/bin:$PATH"
[[ -d "$XDG_DATA_HOME/cargo/bin" ]] && [[ ":$PATH:" != *"$XDG_DATA_HOME/cargo/bin"* ]] && PATH="$PATH:$XDG_DATA_HOME/cargo/bin"
[[ -d "$HOME/go/bin" ]] && [[ ":$PATH:" != *"$HOME/go/bin"* ]] && PATH="$PATH:$HOME/go/bin"

# User specific aliases and functions
# To add directories to your PATH or define additional environment variables, place those changes in .bash_profile (or the equivalent, according to your distribution; for example, Ubuntu uses .profile). For everything else, place the changes in .bashrc.

## tidy up $HOME directory
export ZDOTDIR="$XDG_CONFIG_HOME/zsh" # set zsh default config directory
export LESSHISTFILE="$XDG_STATE_HOME/less/history" # change ~/.lesshst (history is a file)
export CARGO_HOME="$XDG_DATA_HOME/cargo" # change ~/.cargo
export RIPGREP_CONFIG_PATH="$XDG_CONFIG_HOME/ripgrep/ripgreprc" # set file path of ripgrep config file
export WAKATIME_HOME="$XDG_DATA_HOME/wakatime" # wakatime home dir
export _ZO_DATA_DIR="$XDG_DATA_HOME/zoxide" # zoxide

## default programs
export EDITOR="hx" # set editor
export BROWSER="brave-browser" # set browser
export MANPAGER='less' # set manpager

## program settings
# zoxide
export _ZO_RESOLVE_SYMLINKS='1'
# ranger
export RANGER_DEVICONS_SEPARATOR="  " # ranger conf - icon and directory name separator
# set zsh-users/zsh-history-substring-search plugin
export HISTORY_SUBSTRING_SEARCH_HIGHLIGHT_FOUND=''
export HISTORY_SUBSTRING_SEARCH_PREFIXED='true'
# fzf settings
#export FZF_DEFAULT_OPTS="--layout=reverse --height 40% --border" # just to make prettier :)
export FZF_DEFAULT_COMMAND='rg --hidden --files' # include hidden directories/files PS: rg = ripgrep
# matlab
export MATLAB_ROOT='/usr/local/MATLAB/R2022a'
#export LD_LIBRARY_PATH='/usr/local/MATLAB/R2022a/bin/glnxa64:/usr/local/MATLAB/R2022a/sys/os/glnxa64:$LD_LIBRARY_PATH'
# cargo
source "/home/tapyu/.local/share/cargo/env"
