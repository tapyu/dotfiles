local fn = vim.fn

-- Automatically install wbthomason/packer.nvim
local install_path = fn.stdpath "data" .. "/site/pack/packer/start/packer.nvim"
if fn.empty(fn.glob(install_path)) > 0 then
  PACKER_BOOTSTRAP = fn.system {
    "git",
    "clone",
    "--depth",
    "1",
    "https://github.com/wbthomason/packer.nvim",
    install_path,
  }
  print "Installing packer close and reopen Neovim..."
  vim.cmd [[packadd packer.nvim]]
end

-- Autocommand that reloads neovim whenever you save the plugins.lua file
-- whenever 'BufWritePost plugins.lua' event happen (writing this file), it sources this files and run PackerSync
vim.cmd [[
  augroup packer_user_config
    autocmd!
    autocmd BufWritePost plugins.lua source <afile> | PackerSync
  augroup end
]]

-- Use a protected call so we don't error out on first use
-- pcall = protected call
-- this line is the same of
-- local packer = require("packer")
-- with an additional of a boolean status of the requesting
local status_ok, packer = pcall(require, "packer")
if not status_ok then
  print("The package request failed")
  return
end

-- Have packer use a popup (floating) window
packer.init {
  display = {
    open_fn = function()
      return require("packer.util").float { border = "rounded" }
    end,
  },
}



-- Install your plugins here --

-- all your data's packages live in $XDG_DATA_HOME/nivm = .local/share/nvim
-- all your nonlazy packages live in $XDG_DATA_HOME/nvim/site/pack/packer/start
-- all your lazy packages live in $XDG_DATA_HOME/nvim/site/pack/packer/opt
--
-- run -> run a command. Any cd is done inside the location of the package (e.g., in markdown-preview, cd is performed inside $XDG_DATA_HOME/nvim/site/pack/packer/opt/markdown-preview.nvim)
-- cmd -> the package only run when you run this command (it becomes a lazy package)
return packer.startup(function(use)
  -- My plugins here
  use "wbthomason/packer.nvim" -- update packer manage itself. Same as if we run :PackerUpdate
  use "nvim-lua/popup.nvim" -- An implementation of the Popup API from vim in Neovim (a tons of plugins require it)
  use "nvim-lua/plenary.nvim" -- Useful lua functions used in lots of plugins
  use {'iamcco/markdown-preview.nvim', run = 'cd app && yarn install', cmd = 'MarkdownPreview', ft = {md}}

  -- Automatically set up your configuration after cloning packer.nvim
  -- Put this at the end after all plugins
  if PACKER_BOOTSTRAP then
    require("packer").sync()
  end
end)

-- PackerStatus -> shows all the package installed
-- PackerSync -> updates and compile everything. It generates the file in ./lua/plugin/packer_complied.lua
