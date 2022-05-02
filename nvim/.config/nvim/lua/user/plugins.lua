local fn = vim.fn

-- Automatically install wbthomason/packer.nvim
local install_path = fn.stdpath("data") .. "/site/pack/packer/start/packer.nvim"
if fn.empty(fn.glob(install_path)) > 0 then
	PACKER_BOOTSTRAP = fn.system({
		"git",
		"clone",
		"--depth",
		"1",
		"https://github.com/wbthomason/packer.nvim",
		install_path,
	})
	print("Installing packer close and reopen Neovim...")
	vim.cmd([[packadd packer.nvim]])
end

-- Autocommand that reloads neovim whenever you save the plugins.lua file
-- whenever 'BufWritePost plugins.lua' event happen (writing this file), it sources this files and run PackerSync
vim.cmd([[
  augroup packer_user_config
    autocmd!
    autocmd BufWritePost plugins.lua source <afile> | PackerSync
  augroup end
]])

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
packer.init({
	display = {
		open_fn = function()
			return require("packer.util").float({ border = "rounded" })
		end,
	},
})

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
	-- 1. prerequisite
	-- basic plugins
	use("wbthomason/packer.nvim") -- update packer manage itself. Same as if we run :PackerUpdate
	use("nvim-lua/popup.nvim") -- An implementation of the Popup API from vim in Neovim (a tons of plugins require it)
	use("nvim-lua/plenary.nvim") -- Useful lua functions used in lots of plugins
	use("kyazdani42/nvim-web-devicons") -- A bunch of devicons used by a lot of other plugins

	-- 2. foundations
	-- Treesitter - Syntax highlighting, that is, provides colors for the programming language's text
	use({ "nvim-treesitter/nvim-treesitter", run = ":TSUpdate" })
	-- use "p00f/nvim-ts-rainbow"
	-- use "nvim-treesitter/playground"

	-- LSP (Language Server Protocol) - provides varies features for a specific programming language. The main is providing diaginostic (erros, warnings, hints, and info) lintering (flag and colorize the code) and documentation. But some LPS programming language also provides other features, such as stalistic lintering
	use("neovim/nvim-lspconfig") -- enable LSP -> the bare bone LPS - here we find/define the keybinds
	use("williamboman/nvim-lsp-installer") -- it bootstraps (and install/remove) all of the LPS plugins for you. It covers a lot of programming languages's LSP plugins
	use("jose-elias-alvarez/null-ls.nvim") -- provides more diaginostic (style linting) and format fixing

	-- gitsigns - git integration
	use("lewis6991/gitsigns.nvim")

	-- Debug Adpter Protocol (DAP) - abstract the way how the debugging support of development tools communicates with debuggers into a protocol. Software Development Tool or a Software Programming Tool is used by the software developers for creating, editing, maintaining, supporting and debugging other applications, frameworks and programs. In this case, the Software Development Tool is neovim, and the neovim DAP abstract the way of how neovim communicates with generics debuggers
	use("mfussenegger/nvim-dap")
	use({ "rcarriga/nvim-dap-ui", requires = { "mfussenegger/nvim-dap" } }) -- A UI for nvim-dap

	-- 3. discoverability, navegation, and visual
	-- Telescope - A fast way to access files
	use("nvim-telescope/telescope.nvim") -- requires ripgrep
	use("nvim-telescope/telescope-media-files.nvim") -- view media files on telescope

  -- spectre - A search and replace tool (it needs ripgrep)
	use({ "nvim-pack/nvim-spectre", requires = { "kyazdani42/nvim-web-devicons" } })

	-- nvim-tree - file explorer
	use("kyazdani42/nvim-tree.lua")

	-- CheatSheet - to remember autofill, bundled cheats for the editor, vim plugins, nerd-fonts, etc
	use({"sudormrfbin/cheatsheet.nvim", requires = { { "nvim-telescope/telescope.nvim" }, { "nvim-lua/popup.nvim" }, { "nvim-lua/plenary.nvim" } },})

	-- statusline
	use({ "nvim-lualine/lualine.nvim", requires = { "kyazdani42/nvim-web-devicons", opt = true } })

	-- Colorschemes
	use("lunarvim/colorschemes") -- A bunch of colorschemes you can try out with :colorscheme
  use('navarasu/onedark.nvim')

	-- bufferline - a pretty nice way to organize buffers, windows, and tabs
	-- buffers -> it is a file that is loaded into memory, you can open as much buffers as you want. It looks like a tabs definition in other IDE's
	-- window -> it is a space in the screen that can load one of the buffers. you can have multiple windows on your screen, with each window showing one of the buffers (loaded files)
	-- tabs -> a different set of windows showing (possibly) different buffers. The available buffers (loaded files) are the same for all tabs, but the windows can show different buffers in each tab
	use("akinsho/bufferline.nvim")

	-- Toggleterm - terminal integration
	use("akinsho/toggleterm.nvim")

  -- startup - alpha
  use {'goolord/alpha-nvim'}
  use({"Shatur/neovim-session-manager", requires = { 'nvim-lua/plenary.nvim' }})

	-- 4. writing performance enhancement
	-- completions plugins
	use("hrsh7th/nvim-cmp") -- The completion plugin
	use("hrsh7th/cmp-buffer") -- buffer completions
	use("hrsh7th/cmp-path") -- path completions
	use("hrsh7th/cmp-cmdline") -- cmdline completions
	use("saadparwaiz1/cmp_luasnip") -- completions for L3MON4D3/LuaSnip snippets
	use("hrsh7th/cmp-nvim-lsp") --  give us the LSP completions
	use("hrsh7th/cmp-nvim-lua")
	-- snippets
	use("L3MON4D3/LuaSnip") --snippet engine
	use("rafamadriz/friendly-snippets") -- a bunch of snippets to use
	-- comment
	use({"numToStr/Comment.nvim", config = function() require("Comment").setup() end,}) -- requires nvim version >= 0.7
	-- autopairs
	use("windwp/nvim-autopairs")
  -- set of operators and textobjects to search/select/edit sandwiched texts.
  use("machakann/vim-sandwich")

  -- 5. Language-based features
	-- markdown preview
	use({ "iamcco/markdown-preview.nvim", run = "cd app && yarn install" })
  -- LaTeX
  use("lervag/vimtex")
  -- julia
  -- use("autozimu/LanguageClient-neovim")
  use("JuliaEditorSupport/julia-vim")

  -- 6. Miscellaneous
	-- Wakatime
	use("wakatime/vim-wakatime")
	use("moll/vim-bbye") -- :Bdelete command is better than :bdelete as it does not close nvim when the last buffer is closed

	-- Automatically set up your configuration after cloning packer.nvim
	-- Put this at the end after all plugins
	if PACKER_BOOTSTRAP then
		require("packer").sync()
	end
end)

-- PackerStatus -> shows all the package installed
-- PackerSync -> updates and compile everything. It generates the file in ./lua/plugin/packer_complied.lua
