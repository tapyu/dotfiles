# dotfiles

You will need `git` and GNU `stow`.

Steps:
1. `git clone https://github.com/tapyu/dotfiles`
1. `cd dotfiles`
1. `stow --verbose=1 --target=${HOME} */` to symlink everything or just select what you want, e.g., `stow --verbose=1 --target=${HOME} zsh`.
1. `shutdown -r now` to reboot your computer.

Set of config file that is meant to be used with [stow](https://www.gnu.org/software/stow/), see [here](https://www.youtube.com/watch?v=90xMTKml9O0&ab_channel=chris%40machine) for more info. 
