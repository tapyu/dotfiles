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

-- all your data's plugins live in $XDG_DATA_HOME/nivm = .local/share/nvim
-- all your nonlazy plugins live in $XDG_DATA_HOME/nvim/site/pack/packer/start
-- all your lazy plugins live in $XDG_DATA_HOME/nvim/site/pack/packer/opt
--
-- run -> run a command. Any cd is done inside the location of the package (e.g., in markdown-preview, cd is performed inside $XDG_DATA_HOME/nvim/site/pack/packer/opt/markdown-preview.nvim)
-- cmd -> the package is only loaded when you run this command (it becomes a lazy plugin)
-- config -> as soons as the plugin is loaded, it runs a function
-- event -> the pligun is only loaded when an event happens (run :help events to see more info) (it becomes a lazy plugin)
return packer.startup(function(use)
  -- My plugins here
  use "wbthomason/packer.nvim" -- update packer manage itself. Same as if we run :PackerUpdate
  use "nvim-lua/popup.nvim" -- An implementation of the Popup API from vim in Neovim (a tons of plugins require it)
  use "nvim-lua/plenary.nvim" -- Useful lua functions used in lots of plugins
  use {'iamcco/markdown-preview.nvim', run = 'cd app && yarn install'}

  -- Colorschemes
  use "lunarvim/colorschemes" -- A bunch of colorschemes you can try out with :colorscheme
  use 'folke/tokyonight.nvim' -- try :colorscheme tokyonight

  -- completions plugins
  use "hrsh7th/nvim-cmp" -- The completion plugin
  use "hrsh7th/cmp-buffer" -- buffer completions
  use "hrsh7th/cmp-path" -- path completions
  use "hrsh7th/cmp-cmdline" -- cmdline completions
  use "saadparwaiz1/cmp_luasnip" -- snippet completions
  use "hrsh7th/cmp-nvim-lsp" --  give us the LSP completions
  use "hrsh7th/cmp-nvim-lua"

  -- snippets
  use "L3MON4D3/LuaSnip" --snippet engine
  use "rafamadriz/friendly-snippets" -- a bunch of snippets to use

  -- LSP
  use "neovim/nvim-lspconfig" -- enable LSP: the bare bone LPS
  use "williamboman/nvim-lsp-installer" -- it bootstraps all of the LPS for you

  -- Telescope
  use "nvim-telescope/telescope.nvim"
  use 'nvim-telescope/telescope-media-files.nvim'

  -- Treesitter - Syntax highlighting
  use {"nvim-treesitter/nvim-treesitter", run = ":TSUpdate"}
  use "p00f/nvim-ts-rainbow"
  use "nvim-treesitter/playground"
  -- autopairs
  use "windwp/nvim-autopairs"

  -- to remember autofill, bundled cheats for the editor, vim plugins, nerd-fonts, etc
  use {"sudormrfbin/cheatsheet.nvim", requires = {{'nvim-telescope/telescope.nvim'}, {'nvim-lua/popup.nvim'}, {'nvim-lua/plenary.nvim'},}}

  -- Disables relative line numbers when they don't make sense, e.g. when entering insert mode-
  --  use "nkakouros-original/numbers.nvim"

  -- comment
  use {"numToStr/Comment.nvim", config = function() require('Comment').setup() end, tag = 'v0.6',} -- this tag does not break for v0.7 nvim users, so stick with it while you don't go to nvim v0.7
--  use 'numToStr/Comment.nvim'

  -- Automatically set up your configuration after cloning packer.nvim
  -- Put this at the end after all plugins
  if PACKER_BOOTSTRAP then
    require("packer").sync()
  end
end)

-- PackerStatus -> shows all the package installed
-- PackerSync -> updates and compile everything. It generates the file in ./lua/plugin/packer_complied.lua
