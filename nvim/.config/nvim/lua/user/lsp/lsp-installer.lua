-- it comes from https://github.com/williamboman/nvim-lsp-installer
local status_ok, lsp_installer = pcall(require, "nvim-lsp-installer")
if not status_ok then
	return
end
-- Register a handler that will be called for each installed server when it's ready (i.e. when installation is finished
-- or if the server is already installed).

-- on_server_ready() is an built-in functions of lps_installer
lsp_installer.on_server_ready(function(server)
  -- (optional) Customize the options passed to the server
    -- if server.name == "tsserver" then
    --     opts.root_dir = function() ... end
    -- end
	local global_opts = { -- these functions are applied to every single language server on the system (when they are called)
		on_attach = require("user.lsp.handlers").on_attach, -- both on_attach and capabilities are functions
		capabilities = require("user.lsp.handlers").capabilities,
	} -- opts is a table of functions
  local specific_opts -- specific options will be a table of the global functions + the particular functions of a given language

  -- server is the input function
	if server.name == "texlab" then
		local texlab_opts = require("user.lsp.settings.texlab")
   -- vim is a global variable
		specific_opts = vim.tbl_deep_extend("force", texlab_opts, global_opts) -- append to global functions the texlab functions
	end

	if server.name == "julials" then
		-- local julials_opts = require("user.lsp.settings.julials")
   -- vim is a global variable
		-- specific_opts = vim.tbl_deep_extend("force", julials_opts, global_opts)
    specific_opts = global_opts
	end

  -- lua's LSP
	 if server.name == "sumneko_lua" then
	 	local sumneko_opts = require("user.lsp.settings.sumneko_lua")
	 	specific_opts = vim.tbl_deep_extend("force", sumneko_opts, global_opts)
	 end

  -- pyhton's LSP
	 if server.name == "pyright" then
	 	local pyright_opts = require("user.lsp.settings.pyright")
	 	specific_opts = vim.tbl_deep_extend("force", pyright_opts, global_opts)
	 end

  -- R's LPS TODO
	--  if server.name == "r_language_server" then
	--  	local r_language_server_opts = require("user.lsp.settings.r_language_server")
	--  	opts = vim.tbl_deep_extend("force", r_language_server_opts, opts)
	--  end

	-- This setup() function is exactly the same as lspconfig's setup function.
	-- Refer to https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md
	server:setup(specific_opts) -- send to the attached server, which depends on the language used, the specific options of that language
end)
