#!/bin/sh

## install zsh ###
if [ $SHELL != "/bin/zsh" ]; then
  apt install -y zsh
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
apt install -y stow # a symlink farm manager
rm -f $HOME/.zprofile # remove dotfiles in $HOME to avoid error TODO: try to not need it
(cd $HOME/git/dotfiles && sudo -u $SUDO_USER stow --target=$HOME --ignore=check_eol_kernel.sh  --ignore=init1_setup.sh  --ignore=init2_install.sh */) # carry out the symlink manager
source ~/.zprofile
[ -f ~/.profile ] && rm ~/.profile && ln -s $HOME/git/dotfiles/home_dotfiles/.zprofile $HOME/.profile

### packages ###
rm -f /etc/apt/preferences.d/nosnap.pref # for linux mint, enable snapd
apt install -y snapd
apt install -y git
# install rust and cargo
apt install -y rust cargo
export PATH="$PATH:/home/tapyu/.cargo/bin"
# helix install
[ ! -d /tmp/helix ] && mkdir /tmp/helix
git clone https://github.com/helix-editor/helix /tmp/helix
(cd /tmp/helix && cargo install --path helix-term)
apt install -y texlive-full # latex
apt install -y fzf
apt install -y github-cli
apt install -y git-delta
apt install -y tmux # terminal multiplexer
apt install -y xclip
apt install -y zoxide
apt install -y python-pip # pip package
apt install -y python3-distutils # python distutils.core
snap install --classic julia # the Julia programming language
apt install -y ranger # terminal file explorer
apt install -y ueberzug # rager dependency: image preview
(cd $HOME/git/dotfiles/ && sudo -u $SUDO_USER git submodule update --init --recursive) # pull ranger plugins
# add julia to path
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/bin/julia ~/.local/bin/julia 
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/share/julia $XDG_DATA_HOME/julia
sudo -u $SUDO_USER ln -s $XDG_DATA_HOME/julia-1.*/etc/julia $XDG_CONFIG_HOME/julia
# alacritty
apt install -y cmake pkg-config libfreetype6-dev libfontconfig1-dev libxcb-xfixes0-dev libxkbcommon-dev python3 # dependencies
sudo -u $SUDO_USER cargo install alacritty
sudo -u $SUDO_USER cargo install lsd
sudo -u $SUDO_USER cargo install bat
# masterpdfeditor
git clone 'https://mpr.makedeb.org/masterpdfeditor-free' /tmp/
(cd /tmp/masterpdfeditor-free/ && makedeb -si)

# pacstall - An AUR-inspired package manager for Ubuntu
bash -c "$(curl -fsSL https://git.io/JsADh || wget -q https://git.io/JsADh -O -)" # pacstall
sudo -u $SUDO_USER pacstall -I vscode-deb

# insync
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys ACCAF35C
## If the previous command did not work, use this instead:
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ACCAF35C
echo "deb http://apt.insync.io/mint vanessa non-free contrib" > /etc/apt/sources.list.d/insync.list
apt update
apt install insync