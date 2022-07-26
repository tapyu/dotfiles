#!/bin/sh

# Suggesteed partitions and sizes
# - The EFI System: -> Mount point: /boot/ (or /boot/efi/) -> Journaling file system:efi -> size: 300MB (at least)
# - Linux Swap -> Mount point: There isn't -> Journaling file system: linuxswap -> size: equal to memory RAM space (at moment, 24GB)
# - The Root Filesystem -> Mount point: / -> Linux FileSystem, type: ext4 -> size: 150GB (at least)
# - The users home directory -> Mount point: /home/ -> Linux FileSystem, type: ext4 -> size: 200GB (at least)
# - The leftover space -> leave as free space (use it as it is required)

### symlink manager ###
sudo -u $USER yay -S --noconfirm stow # a symlink farm manager
sudo -u $USER (cd $HOME/git/dotfiles && stow */) # carry out the symlink manager

### tidy up - creating important directories ###
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/} ] && sudo -u $USER mkdir $HOME/.local/state # create path to $XDG_STATE_HOME
[ ! -d ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh ] && sudo -u $USER mkdir -p ${XDG_STATE_HOME:-$HOME/.local/state/}/zsh/ # create path to $HISTFILE

### download and installs ###
# Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}
do
  wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf
done

# one-line packages
pacman -S --noconfirm texlive-most texlive-lang texlive-bibtexextra texlive-fontsextra biber # latex files
pacman -S --noconfirm curl # download from url
pacman -S --noconfirm rar # rar and unrar(?) programs
pacman -S --noconfirm github-cli # git CLI to make token password persistent
pacman -S --noconfirm git-delta # dandavison/delta - help to improve git diff and diff commands
pacman -S --noconfirm mlocate # a newer verion of locate command
pacman -S --noconfirm bat # cat with syntax highlighting
pacman -S --noconfirm lsd # the next gen ls command
pacman -S --noconfirm telegram-desktop whatsapp-web-jak # telegram and whatsapp
pacman -S --noconfirm bottom # a process management better than htop (call it with btm)
pacman -S --noconfirm obs-studio # screen recoder
pacman -S --noconfirm fzf # fuzzy finder
pacman -S --noconfirm ripgrep # a replacment for grep
pacman -S --noconfirm brave # web browser
pacman -S --noconfirm vlc # video viewer
pacman -S --noconfirm peek # .gif screen recoder
pacman -S --noconfirm nemo # GUI file explorer
pacman -S --noconfirm slop screenkey ttf-font-awesome # slop -> allows to select windows and/or drag over the desired region interactively without the need of calculating the coordinates manually; screenkey -> keystrokes recoder; ttf-font-awesome -> to enable nice Tux and Win icons
pacman -S --noconfirm xclip # interface to X selections ("the clipboard") from the command line on system with an X11 implementation.
pacman -S --noconfirm python-pip # pip command
pacman -S frei0r-plugins breeze # kdenlive dependencies
pacman -S audacity # audacity - audio recorder
sudo -u $USER pip install youtube-dl # download from youtube
sudo -u $USER pip install trash-cli # trash-cli for KDE, GNOME, and XFCE
# ranger and its dependencies
pacman -S --noconfirm ranger
pacman -S --noconfirm w3m # `w3mimgdisplay` -> for image previews
sudo -u $USER (cd $HOME/git/dotfiles/ && git submodule update --init --recursive) # pull ranger plugins
# Rust
sudo -u $USER curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo -u $USER rustup override set stable
sudo -u $USER rustup update stable
# alacritty
pacman -S --noconfirm cmake freetype2 fontconfig pkg-config make libxcb libxkbcommon python # dependencies
sudo -u $USER cargo install alacritty
# sublime text
curl -O https://download.sublimetext.com/sublimehq-pub.gpg && pacman-key --add sublimehq-pub.gpg && pacman-key --lsign-key 8A8F901A && rm sublimehq-pub.gpg
echo -e "\n[sublime-text]\nServer = https://download.sublimetext.com/arch/stable/x86_64" | sudo tee -a /etc/pacman.conf
pacman --noconfirm -Syu sublime-text
# zoxide
sudo -u $USER curl -sS https://webinstall.dev/zoxide | bash
# yay
sudo -u $USER git clone https://aur.archlinux.org/yay-git.git /tmp/yay-git/
sudo -u $USER (cd /tmp/yay-git && makepkg -si)
sudo -u $USER yay -S --noconfirm masterpdfeditor # pdf reader and editor
sudo -u $USER yay -S --noconfirm radarr # a movie collection manager for Usenet and BitTorrent users
sudo systemctl enable --now radarr
sudo systemctl daemon-reload
sudo -u $USER yay -S --noconfirm ripgrep-all # pdf ripgrep
sudo -u $USER yay -S --noconfirm insync # google drive sync
sudo -u $USER yay -S --noconfirm visual-studio-code-bin # visual studio code
sudo -u $USER yay -S kdenlive-git # kdenlive - video editor

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
sudo -u $USER sed -Ei 's/(^application\/pdf=).*/\1masterpdfeditor5.desktop/' ~/.config/mimeapps.list # masterpdfeditor5 to default pdf
sudo -u $USER gh auth login # github

## GNOME settings
gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # turn alacritty the default terminal emulator
gsettings set org.gnome.desktop.sound allow-volume-above-100-percent 'true' # In Gnome, enable overamplification https://www.reddit.com/r/gnome/comments/exfhc4/overamplification_extension/fgbf9j2/?utm_source=share&utm_medium=web2x&context=3
