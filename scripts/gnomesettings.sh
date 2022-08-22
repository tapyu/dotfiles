#!/bin/sh

### GNOME settings ###
sudo -u $SUDO_USER gsettings set org.gnome.desktop.default-applications.terminal exec /usr/local/bin/alacritty # turn alacritty the default terminal emulator
sudo -u $SUDO_USER gsettings set org.gnome.desktop.sound allow-volume-above-100-percent 'true' # In Gnome, enable overamplification https://www.reddit.com/r/gnome/comments/exfhc4/overamplification_extension/fgbf9j2/?utm_source=share&utm_medium=web2x&context=3 

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