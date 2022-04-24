local status_ok, lspconfig = pcall(require, "lspconfig")
if not status_ok then
  print('lspconfig could not be loaded!')
	return
end

-- require("user.lsp.lsp-installer")
-- require("user.lsp.handlers").setup()
-- require("user.lsp.null-ls")

lspconfig.pyright.setup{}
lspconfig.julials.setup{}
lspconfig.sumneko_lua.setup{}
lspconfig.texlab.setup{}
