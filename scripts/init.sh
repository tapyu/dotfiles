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

### download and installs ###
# Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}; do
  [ ! -f "$HOME/.local/share/fonts/Meslo LG M DZ $(echo $name | sed -r 's/-/ /g') Nerd Font Complete.ttf" ] && wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf
done

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
pacman -S --needed --noconfirm fzf # fuzzy finder
pacman -S --needed --noconfirm ripgrep # a replacment for grep
pacman -S --needed --noconfirm brave-browser # web browser
pacman -S --needed --noconfirm vlc # video viewer
pacman -S --needed --noconfirm peek # .gif screen recoder
pacman -S --needed --noconfirm nemo # GUI file explorer
pacman -S --needed --noconfirm slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons
pacman -S --needed --noconfirm xclip # interface to X selections ("the clipboard") from the command line on system with an X11 implementation.
pacman -S --needed --noconfirm python-pip # pip command
pacman -S --needed --noconfirm frei0r-plugins breeze # kdenlive dependencies
pacman -S --needed --noconfirm audacity # audacity - audio recorder
sudo -u $SUDO_USER pip install youtube-dl # download from youtube
sudo -u $SUDO_USER pip install trash-cli # trash-cli for KDE, GNOME, and XFCE
# ranger and its dependencies
pacman -S --needed --noconfirm ranger
pacman -S --needed --noconfirm w3m # `w3mimgdisplay` -> for image previews
(cd $HOME/git/dotfiles/ && sudo -u $SUDO_USER git submodule update --init --recursive) # pull ranger plugins
# Rust
sudo -u $SUDO_USER curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup override set stable
rustup update stable
# alacritty
pacman -S --needed --noconfirm cmake freetype2 fontconfig pkg-config make libxcb libxkbcommon python # dependencies
cargo install alacritty
# sublime text
curl -O https://download.sublimetext.com/sublimehq-pub.gpg && pacman-key --add sublimehq-pub.gpg && pacman-key --lsign-key 8A8F901A && rm sublimehq-pub.gpg
echo -e "\n[sublime-text]\nServer = https://download.sublimetext.com/arch/stable/x86_64" | sudo tee -a /etc/pacman.conf
pacman --noconfirm -Syu sublime-text
# zoxide
sudo -u $SUDO_USER curl -sS https://webinstall.dev/zoxide | bash
# yay
sudo -u $SUDO_USER git clone https://aur.archlinux.org/yay-git.git /tmp/yay-git/
(cd /tmp/yay-git && sudo -u $SUDO_USER makepkg -si)
sudo -u $SUDO_USER yay -S --noconfirm masterpdfeditor # pdf reader and editor
sudo -u $SUDO_USER yay -S --noconfirm radarr # a movie collection manager for Usenet and BitTorrent users
sudo systemctl enable --now radarr
sudo systemctl daemon-reload
sudo -u $SUDO_USER yay -S --noconfirm insync # google drive sync
sudo -u $SUDO_USER yay -S --noconfirm visual-studio-code-bin # visual studio code
sudo -u $SUDO_USER yay -S --noconfirm kdenlive-git # kdenlive - video editor

### stow - symlink manager ###
pacman --needed -S --noconfirm stow # a symlink farm manager
(cd $HOME/git/dotfiles && sudo -u $SUDO_USER stow --dir=$HOME --ignore=scripts *) # carry out the symlink manager

### GNOME extensions ###
# extensions.gnome.org/extension/517/caffeine/
# shell version: 42 extension verion: 41
wget -P /tmp https://extensions.gnome.org/extension-data/caffeinepatapon.info.v41.shell-extension.zip
gnome-extensions install /tmp/caffeinepatapon.info.v41.shell-extension.zip

# extensions.gnome.org/extension/545/hide-top-bar/
# shell version: 42 extension verion: 107
wget -P /tmp https://extensions.gnome.org/extension-data/hidetopbarmathieu.bidon.ca.v107.shell-extension.zip
gnome-extensions install /tmp/hidetopbarmathieu.bidon.ca.v107.shell-extension.zip

# extensions.gnome.org/extension/615/appindicator-support/
# shell version: 42 extension verion: 42
wget -P /tmp https://extensions.gnome.org/extension-data/appindicatorsupportrgcjonas.gmail.com.v42.shell-extension.zip
gnome-extensions install /tmp/appindicatorsupportrgcjonas.gmail.com.v42.shell-extension.zip

# extensions.gnome.org/extension/1732/gtk-title-bar/
# shell version: 42 extension verion: 10
wget -P /tmp https://extensions.gnome.org/extension-data/gtktitlebarvelitasali.github.io.v10.shell-extension.zip
gnome-extensions install /tmp/gtktitlebarvelitasali.github.io.v10.shell-extension.zip

### setting up configs and default default applications ###
sudo -u $SUDO_USER sed -Ei 's/(^application\/pdf=).*/\1masterpdfeditor5.desktop/' ~/.config/mimeapps.list # masterpdfeditor5 to default pdf
sudo -u $SUDO_USER gh auth login # github

### GNOME settings ###
sudo -u $SUDO_USER gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # turn alacritty the default terminal emulator
sudo -u $SUDO_USER gsettings set org.gnome.desktop.sound allow-volume-above-100-percent 'true' # In Gnome, enable overamplification https://www.reddit.com/r/gnome/comments/exfhc4/overamplification_extension/fgbf9j2/?utm_source=share&utm_medium=web2x&context=3 