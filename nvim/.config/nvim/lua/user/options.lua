-- :help options
local options = {
  backup = false,                          -- creates a backup file
  clipboard = "unnamedplus",               -- allows neovim to access the system clipboard
  cmdheight = 2,                           -- more space in the neovim command line for displaying messages
  completeopt = { "menuone", "noselect", "noinsert" }, -- mostly just for cmp. Watch it to see the diffs amongs these setting: https://www.youtube.com/watch?v=-3S4xVDpLzI
  conceallevel = 0,                        -- so that `` is visible in markdown files
  fileencoding = "utf-8",                  -- the encoding written to a file
  hlsearch = false,                        -- don't highlight all matches on previous search pattern
  hidden = true,                           -- modified buffers are kept in their current state when you leave them, so they are — technically — never 'abandoned'. You will still get a warning if you try to quit Vim with unsaved changes.
  incsearch = true,                        -- highlight matching as type in search
  ignorecase = true,                       -- ignore case in search patterns
  smartcase = true,                        -- smart case
  mouse = "a",                             -- allow the mouse to be used in neovim
  pumheight = 10,                          -- pop up menu height
  showmode = false,                        -- we don't need to see things like -- INSERT -- anymore
  showtabline = 2,                         -- always show tabs
  smartindent = true,                      -- make indenting smarter again
  splitbelow = true,                       -- force all horizontal splits to go below current window
  splitright = true,                       -- force all vertical splits to go to the right of current window
  swapfile = false,                        -- do not create a swapfile
  termguicolors = true,                    -- set term gui colors (most terminals support this) (it appears more :colorscheme options)
  timeoutlen = 1000,                       -- time to wait for a mapped sequence to complete (in milliseconds)
  undofile = true,                         -- enable persistent undo
  updatetime = 300,                        -- faster completion (4000ms default)
  writebackup = false,                     -- if a file is being edited by another program (or was written to file while editing with another program), it is not allowed to be edited
  expandtab = true,                        -- insert space instead tab in insert mode. To insert a real tab when 'expandtab' is on, use CTRL-V<Tab>
  shiftwidth = 2,                          -- the number of spaces inserted for each indentation
  tabstop = 2,                             -- insert 2 spaces for a tab
  cursorline = true,                       -- highlight the current line
  number = true,                           -- set numbered lines
  relativenumber = true,                   -- make other line relative to the current one
  numberwidth = 2,                         -- set number column width to 2 {default 4}
  signcolumn = "yes",                      -- always show the sign column (left thin column used, for instance, for debugging), otherwise it would shift the text each time
  wrap = false,                            -- display lines as one long line
  scrolloff = 8,                           -- Minimal number of screen lines to keep above and below the cursor
  sidescrolloff = 8,
  guifont = "monospace:h10",               -- the font used in graphical neovim applications -> font:height
}

for k, v in pairs(options) do
  vim.opt[k] = v -- apply all options
end

vim.opt.shortmess:append "c"
vim.cmd "set whichwrap+=<,>,[,],h,l"
vim.cmd [[set iskeyword+=-]] -- consider that - is a word character
vim.cmd [[set formatoptions-=cro]] -- TODO: this doesn't seem to work
