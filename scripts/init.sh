#!/bin/sh

# Suggesteed partitions and sizes
# /boot/efi (EFI System, type:efi) -> 300MB (at least)
# Linux Swap (type:swap) -> equal to memory RAM space
# / (Linux FileSystem, type: ext4) -> at 50GB
# /usr/ (type: ext4) -> at least 100 GB
# /home/ (type: ext4) -> at least 200 GB
# the leftover space -> leave as free space (use it as it is required)

# On Ubuntu, you should choose the followig extensions -> gtile, clipboard indicator, workspace switch wraparound, Unite, Hide Top Bar
# path that I need to save on dotfiles repo (get inspired in this repo -> https://github.com/Mach-OS/Machfiles):

## tidy up - creating important directories
[! -d $HOME/.local/state/ ] && mkdir $HOME/.local/state # the $XDG_STATE_HOME (is it necessary?)
[! -d $HOME/.local/state ] && mkdir -p $ZDOTDIR ~/.cache/zsh/ # is it necessary?


# download Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}; do wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf; done

# what I need to download again manually:
# https://trello.com/c/3uugZkiB/99-linux-setup-improvements

# one-line packages
apt-get install terminator texlive-full snapd htop r-base obs-studio variety neovim telegram-desktop git psensor screenfetch rar unrar gparted gimp ocrmypdf pdfgrep flatpak nemo ranger caca-utils curl bat zsh powerline fzf mlocate peek pandoc

pacman -S texlive-most texlive-lang texlive-bibtexextra texlive-fontsextra biber # latex files
pacman -S github-cli # git CLI to make token password persistent
pacman -S git-delta # dandavison/delta - help to improve git diff and diff commands
pacman -S mlocate # a newer verion of locate command
pacman -S rar # rar program (e to extract)
pacman -S bat # cat with syntax highlighting
pacman -S telegram-desktop whatsapp-web-jak # telegram and whatsapp
pacman -S obs-studio # screen recoder
pacman -S fzf # fuzzy finder
pacman -S peek # gif screen recoder
pacman -S slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons


# python packages
apt-get install python3-pip # install pip
apt-get install python3-tk # GUI
# ubuntu packages
apt-get install gnome-tweak-tool dconf-editor
ubuntu-drivers autoinstall # installing third packages
# snap packages
snap install --classic code
snap install discord whatsapp-for-linux lsd
# sublime
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add - # authentication key add
apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list # add to the source list (add repositories)
apt-get update
apt-get install sublime-text
# screen key
sudo add-apt-repository ppa:atareao/atareao # ppa add
sudo apt install screenkeyfk
# kdenlive
#flatpak remote-add --if-not-exists kdeapps --from https://distribute.kde.org/kdeapps.flatpakrepo
#flatpak install kdeapps org.kde.kdenlive
#flatpak update
# 1 -download the flatpak from the website
flatpak install ./org.kde.kdenlive.flatpakref
# audio recoder
sudo apt-add-repository ppa:audio-recorder/ppa
sudo apt-get update
sudo apt-get install audio-recorder
# kitty
curl -L https://sw.kovidgoyal.net/kitty/installer.sh | sh /dev/stdin
ln -s ~/.local/kitty.app/bin/kitty ~/.local/bin/ # Create a symbolic link to add kitty to PATH (assuming ~/.local/bin is in your PATH)
cp ~/.local/kitty.app/share/applications/kitty.desktop ~/.local/share/applications/ # Place the kitty.desktop file somewhere it can be found by the OS
sed -i "s|Icon=kitty|Icon=/home/$USER/.local/kitty.app/share/icons/hicolor/256x256/apps/kitty.png|g" ~/.local/share/applications/kitty.desktop # Update the path to the kitty icon in the kitty.desktop file
gsettings set org.gnome.desktop.default-applications.terminal exec 'kitty' # turn kitty the default application
# youtube-dl for Ubuntu
sudo pip3 install youtube-dl
# pip install packages
pip install trash-cli # trash cli command

# alacritty
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # installing Rust
rustup override set stable
rustup update stable
apt-get install cmake pkg-config libfreetype6-dev libfontconfig1-dev libxcb-xfixes0-dev libxkbcommon-dev python3 # Ubuntu dependencies
# OBS1: see https://github.com/alacritty/alacritty/blob/master/INSTALL.md#prerequisites to install the rest of alacritty
# OBS2: try to create a symbolic link as done with kitty
# turn alacritty the default terminal emulator
gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty

# zoxide install
cargo install zoxide --locked

# nvim requirements check if all requirments are checked with :checkhealth
# for more info, see https://github.com/LunarVim/Neovim-from-scratch/blob/master/README.md
sudo apt install xsel
pip install pynvim
sudo apt-get install npm
sudo npm i -g neovim
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output /tmp/get-pip.py
sudo python2 /tmp/get-pip.py # intall pip for python2
python2 -m pip install neovim

# nvim plugin necessities
npm install --global yarn # a JavaScript package manager compatible with npm, used by iamcco/markdown-preview.nvim
