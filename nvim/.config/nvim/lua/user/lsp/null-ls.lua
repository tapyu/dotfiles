local null_ls_status_ok, null_ls = pcall(require, "null-ls")
if not null_ls_status_ok then
  return
end

-- https://github.com/jose-elias-alvarez/null-ls.nvim/tree/main/lua/null-ls/builtins/formatting
local formatting = null_ls.builtins.formatting
-- https://github.com/jose-elias-alvarez/null-ls.nvim/tree/main/lua/null-ls/builtins/diagnostics
local diagnostics = null_ls.builtins.diagnostics

null_ls.setup {
  debug = false,
  sources = {
    formatting.prettier.with { extra_args = {} }, -- for markdown, yaml and other useless filetype (for me), such as javascript
    formatting.black.with { extra_args = { "--fast" } },
    -- formatting.yapf, -- style linting for python
    formatting.stylua, -- style linting for lua
    diagnostics.flake8, -- style linting for python
  },
}
