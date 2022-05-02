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
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/} ] && mkdir $HOME/.local/state # create path to $XDG_STATE_HOME
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh ] && mkdir -p ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh/ # create path to $HISTFILE


# download Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}
do
  # wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf
  echo test
done

# what I need to download again manually:
# https://trello.com/c/3uugZkiB/99-linux-setup-improvements

# one-line packages
# apt-get install terminator texlive-full snapd htop r-base obs-studio variety neovim telegram-desktop git psensor screenfetch rar unrar gparted gimp ocrmypdf pdfgrep flatpak nemo ranger caca-utils curl bat zsh powerline fzf mlocate peek pandoc

pacman -S --noconfirm texlive-most texlive-lang texlive-bibtexextra texlive-fontsextra biber # latex files
pacman -S --noconfirm curl # download from url
pacman -S --noconfirm rar # rar program (e option to extract)
pacman -S --noconfirm github-cli # git CLI to make token password persistent
pacman -S --noconfirm git-delta # dandavison/delta - help to improve git diff and diff commands
pacman -S --noconfirm mlocate # a newer verion of locate command
pacman -S --noconfirm bat # cat with syntax highlighting
pacman -S --noconfirm telegram-desktop whatsapp-web-jak # telegram and whatsapp
pacman -S --noconfirm obs-studio # screen recoder
pacman -S --noconfirm fzf # fuzzy finder
pacman -S --noconfirm peek # gif screen recoder
pacman -S --noconfirm slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons
pacman -S --noconfirm xclip # interface to X selections ("the clipboard") from the command line on system with an X11 implementation.
pacman -S --noconfirm python-pip # pip command
# Rust
sudo -u $USER curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo -u $USER rustup override set stable
sudo -u $USER rustup update stable
# alacritty
pacman -S --noconfirm cmake freetype2 fontconfig pkg-config make libxcb libxkbcommon python # dependencies
sudo -u $USER cargo install alacritty
gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # turn alacritty the default terminal emulator
# sublime text
curl -O https://download.sublimetext.com/sublimehq-pub.gpg && sudo pacman-key --add sublimehq-pub.gpg && sudo pacman-key --lsign-key 8A8F901A && rm sublimehq-pub.gpg
echo -e "\n[sublime-text]\nServer = https://download.sublimetext.com/arch/stable/x86_64" | sudo tee -a /etc/pacman.conf
pacman --noconfirm -Syu sublime-text
# zoxide install
sudo -u $USER curl -sS https://webinstall.dev/zoxide | bash

# TODO: create symlinks

# setting up github
sudo -u $USER gh auth login
