#!/bin/sh

### installs ###
# one-line packages
pacman -S --needed --noconfirm texlive-most texlive-lang texlive-bibtexextra texlive-fontsextra biber # latex files
pacman -S --needed --noconfirm curl # download from url
pacman -S --needed --noconfirm rar # rar and unrar(?) programs
pacman -S --needed --noconfirm github-cli # git CLI to make token password persistent
pacman -S --needed --noconfirm git-delta # dandavison/delta - help to improve git diff and diff commands
pacman -S --needed --noconfirm mlocate # a newer verion of locate command
pacman -S --needed --noconfirm bat # cat with syntax highlighting
pacman -S --needed --noconfirm lsd # the next gen ls command
pacman -S --needed --noconfirm telegram-desktop whatsapp-web-jak # telegram and whatsapp
pacman -S --needed --noconfirm bottom # a process management better than htop (call it with btm)
pacman -S --needed --noconfirm obs-studio # screen recoder
pacman -S --needed --noconfirm helix # A post-modern modal text editor.
pacman -S --needed --noconfirm fzf # fuzzy finder
pacman -S --needed --noconfirm ripgrep # a replacment for grep
pacman -S --needed --noconfirm brave-browser # web browser
pacman -S --needed --noconfirm vlc # video viewer
pacman -S --needed --noconfirm tmux # terminal multiplexer
pacman -S --needed --noconfirm peek # .gif screen recoder
pacman -S --needed --noconfirm nemo # GUI file explorer
pacman -S --needed --noconfirm slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons
pacman -S --needed --noconfirm xclip # interface to X selections ("the clipboard") from the command line on system with an X11 implementation.
pacman -S --needed --noconfirm python-pip # pip command
pacman -S --needed --noconfirm frei0r-plugins breeze # kdenlive dependencies
pacman -S --needed --noconfirm audacity # audacity - audio recorder
pacman -S --needed --noconfirm julia # The Julia Programming Language
# add julia to path
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/bin/julia ~/.local/bin/julia 
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/share/julia $XDG_DATA_HOME/julia
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/etc/julia $XDG_CONFIG_HOME/julia

sudo -u $SUDO_USER pip install youtube-dl # download from youtube
sudo -u $SUDO_USER pip install trash-cli # trash-cli for KDE, GNOME, and XFCE
# ranger and its dependencies
pacman -S --needed --noconfirm ranger
pacman -S --needed --noconfirm w3m # `w3mimgdisplay` -> for image previews
(cd $HOME/git/dotfiles/ && sudo -u $SUDO_USER git submodule update --init --recursive) # pull ranger plugins
# Rust and cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sudo -u $SUDO_USER sh
sudo -u $SUDO_USER rustup override set stable
sudo -u $SUDO_USER rustup update stable
rm -f .profile # cargo creates a .profile file, if must be deleted to create to create a symlink to .zprofile
sudo -u $SUDO_USER ln -s $HOME/.zprofile $HOME/.profile # .profile is what is really sourced at X11 session
# alacritty
pacman -S --needed --noconfirm cmake freetype2 fontconfig pkg-config make libxcb libxkbcommon python # dependencies
sudo -u $SUDO_USER cargo install alacritty
# zoxide
sudo -u $SUDO_USER curl -sS https://webinstall.dev/zoxide | bash
# yay
sudo -u $SUDO_USER git clone https://aur.archlinux.org/yay-git.git /tmp/yay-git/
(cd /tmp/yay-git && sudo -u $SUDO_USER makepkg -si)
sudo -u $SUDO_USER yay -S --needed --noconfirm masterpdfeditor # pdf reader and editor
sudo -u $SUDO_USER yay -S --needed --noconfirm insync # google drive sync
sudo -u $SUDO_USER yay -S --needed --noconfirm visual-studio-code-bin # visual studio code
sudo -u $SUDO_USER yay -S --needed --noconfirm kdenlive-git # kdenlive - video editor

### GNOME extensions ###
# extensions.gnome.org/extension/517/caffeine/
# shell version: 42 extension verion: 41
sudo -u $SUDO_USER wget -P /tmp https://extensions.gnome.org/extension-data/caffeinepatapon.info.v41.shell-extension.zip
sudo -u $SUDO_USER gnome-extensions install /tmp/caffeinepatapon.info.v41.shell-extension.zip

# extensions.gnome.org/extension/545/hide-top-bar/
# shell version: 42 extension verion: 107
sudo -u $SUDO_USER wget -P /tmp https://extensions.gnome.org/extension-data/hidetopbarmathieu.bidon.ca.v107.shell-extension.zip
sudo -u $SUDO_USER gnome-extensions install /tmp/hidetopbarmathieu.bidon.ca.v107.shell-extension.zip

# extensions.gnome.org/extension/615/appindicator-support/
# shell version: 42 extension verion: 42
sudo -u $SUDO_USER wget -P /tmp https://extensions.gnome.org/extension-data/appindicatorsupportrgcjonas.gmail.com.v42.shell-extension.zip
sudo -u $SUDO_USER gnome-extensions install /tmp/appindicatorsupportrgcjonas.gmail.com.v42.shell-extension.zip

# extensions.gnome.org/extension/1732/gtk-title-bar/
# shell version: 42 extension verion: 10
sudo -u $SUDO_USER wget -P /tmp https://extensions.gnome.org/extension-data/gtktitlebarvelitasali.github.io.v10.shell-extension.zip
sudo -u $SUDO_USER gnome-extensions install /tmp/gtktitlebarvelitasali.github.io.v10.shell-extension.zip

### setting up configs and default default applications ###
sudo -u $SUDO_USER sed -Ei 's/(^application\/pdf=).*/\1masterpdfeditor5.desktop/' ~/.config/mimeapps.list # masterpdfeditor5 to default pdf
sudo -u $SUDO_USER gh auth login # github

### GNOME settings ###
sudo -u $SUDO_USER gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # turn alacritty the default terminal emulator
sudo -u $SUDO_USER gsettings set org.gnome.desktop.sound allow-volume-above-100-percent 'true' # In Gnome, enable overamplification https://www.reddit.com/r/gnome/comments/exfhc4/overamplification_extension/fgbf9j2/?utm_source=share&utm_medium=web2x&context=3 

### GNOME gsettings shortcut ###
# get more ideas here: https://blog.programster.org/using-the-cli-to-set-custom-keyboard-shortcuts
# write your solution here: https://askubuntu.com/questions/597395/how-to-set-custom-keyboard-shortcuts-from-terminal
KEY_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
"['$KEY_PATH/custom0/']" # add more into this list for more shortcuts

# Terminal
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/ name "Terminal"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/ command "'/home/tapyu/.local/share/cargo/bin/alacritty'"
gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/  binding "<Control><Alt>t"

# TODO:  add keyboard
