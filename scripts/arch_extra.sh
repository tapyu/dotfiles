#!/bin/sh

### installs ###
pacman -S --needed --noconfirm curl # download from url
pacman -S --needed --noconfirm rar # rar and unrar(?) programs
pacman -S --needed --noconfirm mlocate # a newer version of locate command
pacman -S --needed --noconfirm bat # cat with syntax highlighting
pacman -S --needed --noconfirm telegram-desktop whatsapp-web-jak # telegram and whatsapp
pacman -S --needed --noconfirm tree # see directories as tree
pacman -S --needed --noconfirm obs-studio # screen recoder
pacman -S --needed --noconfirm ripgrep # a replacment for grep
pacman -S --needed --noconfirm pdfgrep # grep for pdf files
pacman -S --needed --noconfirm vlc # video viewer
pacman -S --needed --noconfirm gimp # image editors
pacman -S --needed --noconfirm peek # .gif screen recoder
pacman -S --needed --noconfirm nemo # GUI file explorer
pacman -S --needed --noconfirm slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons
pacman -S --needed --noconfirm frei0r-plugins breeze # kdenlive dependencies
pacman -S --needed --noconfirm audacity # audacity - audio recorder

sudo -u $SUDO_USER pip install youtube-dl # download from youtube
sudo -u $SUDO_USER pip install trash-cli # trash-cli for KDE, GNOME, and XFCE
# zoxide
sudo -u $SUDO_USER curl -sS https://webinstall.dev/zoxide | bash
sudo -u $SUDO_USER yay -S --needed --noconfirm insync # google drive sync
sudo -u $SUDO_USER yay -S --needed --noconfirm kdenlive-git # kdenlive - video editor
