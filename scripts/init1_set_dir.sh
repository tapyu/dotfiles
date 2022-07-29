#!/bin/sh

# Suggesteed partitions and sizes
# - The EFI System: -> Mount point: /boot/efi/ -> Journaling file system:efi -> size: 300MB (at least)
# - Linux Swap -> Mount point: There isn't -> Journaling file system: linuxswap -> size: equal to memory RAM space (at moment, 24GB)
# - The Root Filesystem -> Mount point: / -> Linux FileSystem, type: ext4 -> size: 150GB (at least)
# - The users home directory -> Mount point: /home/ -> Linux FileSystem, type: ext4 -> size: 200GB (at least)
# - The leftover space -> leave as free space (use it as it is required)

### symlink manager ###
sudo -u $SUDO_USER pacman --needed -S --noconfirm stow # a symlink farm manager
sudo -u $SUDO_USER (cd $HOME/git/dotfiles && stow */) # carry out the symlink manager

### tidy up - creating important directories ###
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/} ] && sudo -u $SUDO_USER mkdir $HOME/.local/state # create path to $XDG_STATE_HOME
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh ] && sudo -u $SUDO_USER mkdir -p ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh/ # create path to $HISTFILE

### download and installs ###
# Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}
do
  wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf
done

