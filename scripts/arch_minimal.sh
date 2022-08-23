#!/bin/sh

# PS: run it as sudo --preserve-env sh init.sh to preserve the variables
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
rm -f $HOME/.zprofile # remove dotfiles in $HOME to avoid error TODO: try to not need it
(cd $HOME/git/dotfiles && sudo -u $SUDO_USER stow --target=$HOME --ignore=check_eol_kernel.sh  --ignore=init1_setup.sh  --ignore=init2_install.sh */) # carry out the symlink manager
source ~/.zprofile

### one-line packages ###
pacman -S --needed --noconfirm texlive-most texlive-lang texlive-bibtexextra texlive-fontsextra biber # latex files
pacman -S --needed --noconfirm github-cli # git CLI to make token password persistent
pacman -S --needed --noconfirm git-delta # dandavison/delta - help to improve git diff and diff commands
pacman -S --needed --noconfirm helix # A post-modern modal text editor
pacman -S --needed --noconfirm fzf # fuzzy finder
# pacman -S --needed --noconfirm brave-browser # web browser (it is available only on Majaro)
pacman -S --needed --noconfirm tmux # terminal multiplexer
pacman -S --needed --noconfirm xclip # interface to X selections ("the clipboard") from the command line on system with an X11 implementation.
pacman -S --needed --noconfirm python-pip # pip command
pacman -S --needed --noconfirm julia # The Julia Programming Language
# ranger and its dependencies
pacman -S --needed --noconfirm ranger
pacman -S --needed --noconfirm w3m ueberzug # ranger image preview default; `w3mimgdisplay` -> for image previews obs:set ueberzug as the default image preview for alacritty!
(cd $HOME/git/dotfiles/ && sudo -u $SUDO_USER git submodule update --init --recursive) # pull ranger plugins
# add julia to path
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/bin/julia ~/.local/bin/julia 
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/share/julia $XDG_DATA_HOME/julia
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/etc/julia $XDG_CONFIG_HOME/julia
# Rust and cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sudo -u $SUDO_USER sh
source "$HOME/.cargo/env"
sudo -u $SUDO_USER rustup override set stable
sudo -u $SUDO_USER rustup update stable
rm -f .profile # cargo creates a .profile file, if must be deleted to create to create a symlink to .zprofile
sudo -u $SUDO_USER ln -s $HOME/.zprofile $HOME/.profile # .profile is what is really sourced at X11 session
# alacritty
pacman -S --needed --noconfirm cmake freetype2 fontconfig pkg-config make libxcb libxkbcommon python # dependencies
sudo -u $SUDO_USER cargo install alacritty
# yay
sudo -u $SUDO_USER git clone https://aur.archlinux.org/yay-git.git /tmp/yay-git/
(cd /tmp/yay-git && sudo -u $SUDO_USER makepkg -si)
sudo -u $SUDO_USER yay -S --needed --noconfirm masterpdfeditor # pdf reader and editor
sudo -u $SUDO_USER yay -S --needed --noconfirm visual-studio-code-bin # visual studio code
sudo -u $SUDO_USER gh auth login # github authentication login
