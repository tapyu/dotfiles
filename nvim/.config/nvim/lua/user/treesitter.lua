--local configs = require("nvim-treesitter.configs")
local status_ok, configs = pcall(require, "nvim-treesitter.configs")
if not status_ok then
  print("nvim-treesitter failed!")
  return
end

configs.setup {
  ensure_installed = "all",
  sync_install = false,
  ignore_install = { "swift", "phpdoc" }, -- List of parsers to ignore installing
  highlight = {
    enable = true, -- false will disable the whole extension
    disable = { "" }, -- list of language that will be disabled
    additional_vim_regex_highlighting = true,

  },
  rainbow = {
    enable = true,
    -- disable = { "jsx", "cpp" }, list of languages you want to disable the plugin for
    extended_mode = true, -- Also highlight non-bracket delimiters like html tags, boolean or table: lang -> boolean
    max_file_lines = nil, -- Do not enable for files with more than n lines, int
    -- colors = {}, -- table of hex strings
    -- termcolors = {} -- table of colour name strings
  },
  indent = { enable = true, disable = { "yaml" } }, -- put the cursor where it is expected to be
  incremental_selection = { -- enter in visual mode and select area based on treesitter
    enable = true,
    keymaps = {
      init_selection = "gnn",
      node_incremental = "grn",
      scope_incremental = "grc",
      node_decremental = "grm",
    },
  },
}

-- enable folding
vim.cmd [[set foldmethod=expr]] -- set the foldmethod to expression
vim.cmd [[set foldexpr=nvim_treesitter#foldexpr()]] -- use foldexpr() function from treesitter fol method to fold
