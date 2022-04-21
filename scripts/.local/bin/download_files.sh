#!/bin/bash

# download Meslo patched Nerd-fonts into ~/.local/share/fonts/
for name in {Regular,Italic,Bold-Italic}; do wget --directory-prefix="$HOME/.local/share/fonts" https://raw.githubusercontents.com/ryanoasis/nerd-fonts/master/patched-fonts/Meslo/M-DZ/$name/complete/Meslo%20LG%20M%20DZ%20$(echo $name | sed -r 's/-/%20/g')%20Nerd%20Font%20Complete.ttf; done
