local colorscheme_name = "darkplus"

local is_status_ok, colorscheme = pcall(require, colorscheme_name)

if not is_status_ok then
  print("colorscheme " .. colorscheme_name .. "not found!")
end

if type(colorscheme) == "boolean" then -- colorscheme does not have lua configs, e.g., darkplus
  vim.cmd("colorscheme " .. colorscheme_name)
else -- colorscheme has lua configs, e.g., onedark
  if colorscheme_name == "onedark" then
    colorscheme.setup{
      style = 'warmer'
    }
  end
  colorscheme.load()
end
