#!/bin/sh

### GNOME default applications ###
sudo -u $SUDO_USER gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # alacritty to default terminal emulator
sudo -u $SUDO_USER sed -Ei 's/(^application\/pdf=).*/\1masterpdfeditor5.desktop/' ~/.config/mimeapps.list # masterpdfeditor5 to default pdf
sudo -u $SUDO_USER gsettings set org.gnome.desktop.sound allow-volume-above-100-percent 'true' # In Gnome, enable overamplification https://www.reddit.com/r/gnome/comments/exfhc4/overamplification_extension/fgbf9j2/?utm_source=share&utm_medium=web2x&context=3 

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

### GNOME gsettings shortcut ###
# get more ideas here: https://blog.programster.org/using-the-cli-to-set-custom-keyboard-shortcuts
# write your solution here: https://askubuntu.com/questions/597395/how-to-set-custom-keyboard-shortcuts-from-terminal
sudo -u $SUDO_USER KEY_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
"['$KEY_PATH/custom0/', '$KEY_PATH/custom1/']" # add more into this list for more shortcuts
# Terminal
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/ name "Terminal"
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/ command "'/home/tapyu/.local/share/cargo/bin/alacritty'"
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom0/  binding "<Control><Alt>t"
# Web Browser
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom1/ name "Brave Browser"
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom1/ command "'/usr/bin/brave'"
sudo -u $SUDO_USER gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEY_PATH/custom1/  binding "<Super>1"

# TODO: add these extensions
# https://extensions.gnome.org/extension/28/gtile/
# https://extensions.gnome.org/extension/4496/hide-panel-light-version-without-hot-corner/

# TODO:  add keyboard