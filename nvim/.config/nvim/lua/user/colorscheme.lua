local colorscheme = "darkplus"

-- pcall -> protected call
-- _ -> ignore the return value
local status_ok, _ = pcall(vim.cmd, "colorscheme " .. colorscheme) -- verify if the selected colorscheme exists
if not status_ok then
  vim.notify("colorscheme " .. colorscheme .. " not found!")
  return
end
