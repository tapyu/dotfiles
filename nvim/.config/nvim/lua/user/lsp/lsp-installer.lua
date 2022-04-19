local status_ok, lsp_installer = pcall(require, "nvim-lsp-installer")
if not status_ok then
	return
end

-- on_server_ready() is an built-in functions of lps_installer
lsp_installer.on_server_ready(function(server)
	local opts = { -- these functions are applied to every single language server on the system
		on_attach = require("user.lsp.handlers").on_attach, -- both on_attach and capabilities are functions
		capabilities = require("user.lsp.handlers").capabilities,
	}

  -- server is the input function
  -- LaTeX's LSP TODO
	-- if server.name == "texlab" then
	-- 	local texlab_opts = require("user.lsp.settings.texlab")
  --  -- vim is a global variable
	-- 	opts = vim.tbl_deep_extend("force", texlab_opts, opts)
	-- end

   -- julia's LSP TODO
	-- if server.name == "julials" then
	-- 	local julials_opts = require("user.lsp.settings.julials")
  --  -- vim is a global variable
	-- 	opts = vim.tbl_deep_extend("force", julials_opts, opts)
	-- end

  -- lua's LSP
	 if server.name == "sumneko_lua" then
	 	local sumneko_opts = require("user.lsp.settings.sumneko_lua")
	 	opts = vim.tbl_deep_extend("force", sumneko_opts, opts)
	 end

  -- pyhton's LSP
	 if server.name == "pyright" then
	 	local pyright_opts = require("user.lsp.settings.pyright")
	 	opts = vim.tbl_deep_extend("force", pyright_opts, opts)
	 end

  -- R's LPS TODO
	--  if server.name == "r_language_server" then
	--  	local r_language_server_opts = require("user.lsp.settings.r_language_server")
	--  	opts = vim.tbl_deep_extend("force", r_language_server_opts, opts)
	--  end

	-- This setup() function is exactly the same as lspconfig's setup function.
	-- Refer to https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md
	server:setup(opts)
end)
