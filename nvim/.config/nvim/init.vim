" default options
set completeopt=menuone,noinsert,noselect " options for insert mode completion (help 'ins-completion' for more info)
" set mouse=a " enable mouse for all modes
set splitright " When on, splitting a window will put the new window right of the current one :vsplit
set splitbelow " When on, splitting a window will put the new window below the current one. :split
set expandtab " insert space instead tab in insert mode. To insert a real tab when 'expandtab' is on, use CTRL-V<Tab>
set tabstop=2 " Number of spaces that a <Tab> in the file counts for
set shiftwidth=2 " Number of spaces to use for each step of (auto)indent
set ignorecase " ignore case sesitive -> Can be overruled by using "\c" or "\C" in the pattern
set smartcase " Override the 'ignorecase' option if the search pattern contains upper case characters.
set incsearch " While typing a search command, show where the pattern, as it was typed so far, matches.  The matched string is highlighted.
set diffopt+=vertical " Start diff mode with vertical splits
set hidden " modified buffers are kept in their current state when you leave them so they are — technically — never 'abandoned'. You will still get a warning if you try to quit Vim with unsaved changes. A buffer is an area of Vim's memory used to hold text read from a file. In addition, an empty buffer with no associated file can be created to allow the entry of text.
set nobackup " do not make a backup before overwriting a file
set nowritebackup " do not make backup while the file is being written
set cmdheight=1 " number of screen lines to use for the command-line
set shortmess+=c " his option helps to avoid all the hit-enter prompts caused by file messages
set signcolumn=yes " when and how to draw the signcolumn (used by the debbuger, for instance)
set updatetime=750
set relativenumber " make other line relative to the current
set number " put number line on the current line

filetype plugin indent on
let mapleader=" " " set leader to space
" normal no replace to the vimrc file
nnoremap <leader>v :e ~/.config/nvim/init.vim<CR>

" enable terminal gui colors
if (has("termguicolors"))
  set termguicolors
endif
