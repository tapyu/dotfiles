#!/bin/sh

# PS: run it as sudo --preserve-env=HOME,SHELL sh init.sh to preserve these variables
# Suggesteed partitions and sizes
# - The EFI System: -> Mount point: /boot/efi/ -> Journaling file system:efi -> size: 300MB (at least)
# - Linux Swap -> Mount point: There isn't -> Journaling file system: linuxswap -> size: equal to memory RAM space (at moment, 24GB)
# - The Root Filesystem -> Mount point: / -> Linux FileSystem, type: ext4 -> size: 150GB (at least)
# - The users home directory -> Mount point: /home/ -> Linux FileSystem, type: ext4 -> size: 200GB (at least)
# - The leftover space -> leave as free space (use it as it is required)

## install zsh ###
if [ $SHELL != "/bin/zsh" ]; then
  pacman -S --needed --no-confirm zsh
  sudo -u $SUDO_USER chsh -s /usr/bin/zsh
fi

### tidy up - creating important directories ###
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/} ] && sudo -u $SUDO_USER mkdir $HOME/.local/state # create path to $XDG_STATE_HOME
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh ] && sudo -u $SUDO_USER mkdir -p ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh/ # create path to $HISTFILE
[ ! -d $HOME/.local/bin ] && sudo -u $SUDO_USER mkdir -p $HOME/.local/bin # create user-specific executable files

### install Meslo patched Nerd-fonts into ~/.local/share/fonts/ ###
for name in {Regular,Italic,Bold-Italic}; do
  [ ! -f "$HOME/.local/share/fonts/Meslo LG M DZ $(echo $name | sed -r 's/-/ /g') Nerd Font Complete.ttf" ] && wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf
done

### stow - symlink manager ###
pacman --needed -S --noconfirm stow # a symlink farm manager
rm $HOME/.zprofile # remove dotfiles in $HOME to avoid error TODO: try to not need it
(cd $HOME/git/dotfiles && sudo -u $SUDO_USER stow --target=$HOME --ignore=scripts */) # carry out the symlink manager
