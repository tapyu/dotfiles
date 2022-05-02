local is_status_ok, alpha = pcall(require, "alpha")

if not is_status_ok then
  print("alpha plugin not found")
  return
end

local dashboard = require("alpha.themes.dashboard")

dashboard.section.buttons.val = {
  dashboard.button("SPC s o", "  Open sessions", "<cmd>SessionManager load_session<CR>"),
  dashboard.button("SPC s l", "  Open last session", "<cmd>SessionManager load_last_session<CR>"),
  dashboard.button("SPC f f", "  Find file", ":Telescope find_files hidden=true no_ignore=true<CR>"),
  dashboard.button("SPC f h", "  Recently opened files", "<cmd>Telescope oldfiles<CR>"),
  -- dashboard.button("SPC f r", "  Frecency/MRU"),
  dashboard.button("SPC f g", "  Find word",  "<cmd>Telescope live_grep<cr>"),
  -- dashboard.button("SPC f m", "  Jump to bookmarks"),
}

-- find more here https://steamcommunity.com/groups/asciiartamalgamation/discussions/5/3008934419468851875/
-- dashboard.section.header.val = {
--   [[ ⠄⠄⠄⠄⠄⣾⣿⣿⣿⣯⣵⣾⣿⣿⡶⠦⠭⢁⠩⢭⣭⣵⣶⣶⡬⣄⣀⡀⠄⠄ ]],
--   [[ ⠄⠄⠄⡀⠘⠻⣿⣿⣿⣿⡿⠟⠩⠶⠚⠻⠟⠳⢶⣮⢫⣥⠶⠒⠒⠒⠒⠆⠐⠒ ]],
--   [[ ⠄⢠⣾⢇⣿⣿⣶⣦⢠⠰⡕⢤⠆⠄⠰⢠⢠⠄⠰⢠⠠⠄⡀⠄⢊⢯⠄⡅⠂⠄ ]],
--   [[ ⢠⣿⣿⣿⣿⣿⣿⣿⣏⠘⢼⠬⠆⠄⢘⠨⢐⠄⢘⠈⣼⡄⠄⠄⡢⡲⠄⠂⠠⠄ ]],
--   [[ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣥⣀⡁⠄⠘⠘⠘⢀⣠⣾⣿⢿⣦⣁⠙⠃⠄⠃⠐⣀ ]],
--   [[ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣋⣵⣾⣿⣿⣿⣿⣦⣀⣶⣾⣿⣿⡉⠉⠉ ]],
--   [[ ⣿⣿⣿⣿⣿⣿⣿⠟⣫⣥⣬⣭⣛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄ ]],
--   [[ ⣿⣿⣿⣿⣿⣿⣿⠸⣿⣏⣙⠿⣿⣿⣶⣦⣍⣙⠿⠿⠿⠿⠿⠿⠿⠿⣛⣩⣶⠄ ]],
--   [[ ⣛⣛⣛⠿⠿⣿⣿⣿⣮⣙⠿⢿⣶⣶⣭⣭⣛⣛⣛⣛⠛⠛⠻⣛⣛⣛⣛⣋⠁⢀ ]],
--   [[ ⣿⣿⣿⣿⣿⣶⣬⢙⡻⠿⠿⣷⣤⣝⣛⣛⣛⣛⣛⣛⣛⣛⠛⠛⣛⣛⠛⣡⣴⣿ ]],
-- }


-- dashboard.section.header.val = {
--   [[ ⠄⠄⠄⠄⣠⣴⣿⣿⣿⣷⣦⡠⣴⣶⣶⣶⣦⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄ ]],
--   [[ ⠄⠄⠄⣴⣿⣿⣫⣭⣭⣭⣭⣥⢹⣟⣛⣛⣛⣃⣀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄ ]],
--   [[ ⠄⣠⢸⣿⣿⣿⣿⢯⡓⢻⠿⠿⠷⡜⣯⠭⢽⠿⠯⠽⣀⠄⠄⠄⠄⠄⠄⠄⠄⠄ ]],
--   [[ ⣼⣿⣾⣿⣿⣿⣥⣝⠂⠐⠈⢸⠿⢆⠱⠯⠄⠈⠸⣛⡒⠄⠄⠄⠄⠄⠄⠄⠄⠄ ]],
--   [[ ⣿⣿⣿⣿⣿⣿⣿⣶⣶⣭⡭⢟⣲⣶⡿⠿⠿⠿⠿⠋⠄⠄⣴⠶⠶⠶⠶⠶⢶⡀ ]],
--   [[ ⣿⣿⣿⣿⣿⢟⣛⠿⢿⣷⣾⣿⣿⣿⣿⣿⣿⣿⣷⡄⠄⢰⠇⠄⠄⠄⠄⠄⠈⣧ ]],
--   [[ ⣿⣿⣿⣿⣷⡹⣭⣛⠳⠶⠬⠭⢭⣝⣛⣛⣛⣫⣭⡥⠄⠸⡄⣶⣶⣾⣿⣿⢇⡟ ]],
--   [[ ⠿⣿⣿⣿⣿⣿⣦⣭⣛⣛⡛⠳⠶⠶⠶⣶⣶⣶⠶⠄⠄⠄⠙⠮⣽⣛⣫⡵⠊⠁ ]],
--   [[ ⣍⡲⠮⣍⣙⣛⣛⡻⠿⠿⠿⠿⠿⠿⠿⠖⠂⠄⠄⠄⠄⠄⠄⠄⠄⣸⠄⠄⠄⠄ ]],
--   [[ ⣿⣿⣿⣶⣦⣬⣭⣭⣭⣝⣭⣭⣭⣴⣷⣦⡀⠄⠄⠄⠄⠄⠄⠠⠤⠿⠦⠤⠄⠄ ]],
-- }

dashboard.section.header.val = {
  [[ ⣿⣿⣿⣿⣿⣿⣿⠛⢩⣴⣶⣶⣶⣌⠙⠫⠛⢋⣭⣤⣤⣤ ]],
  [[ ⣿⣿⣿⣿⣿⡟⢡⣾⣿⠿⣛⣛⣛⣛⣛⡳⠆⢻⣿⣿⣿⠿⠿⠷⡌ ]],
  [[ ⣿⣿⣿⣿⠏⣰⣿⣿⣴⣿⣿⣿⡿⠟⠛⠛⠒⠄⢶⣶⣶⣾⡿⠶⠒⠲⠌ ]],
  [[ ⣿⣿⠏⣡⢨⣝⡻⠿⣿⢛⣩⡵⠞⡫⠭⠭⣭⠭⠤⠈⠭⠒⣒⠩⠭⠭⣍⠒⠈ ]],
  [[ ⡿⢁⣾⣿⣸⣿⣿⣷⣬⡉⠁⠄⠁⠄⠄⠄⠄⠄⠄⠄⣶⠄⠄⠄⠄⠄⠄⠄⠄⢀ ]],
  [[ ⢡⣾⣿⣿⣿⣿⣿⣿⣿⣧⡀⠄⠄⠄⠄⠄⠄⠄⢀⣠⣿⣦⣤⣀⣀⣀⣀⠄ ]],
  [[ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⡶⢇⣰⣿⣿⣟⠿⠿⠿⠿⠟ ]],
  [[ ⣿⣿⣿⣿⣿⣿⣿⡟⢛⡛⠿⠿⣿⣧⣶⣶⣿⣿⣿⣿⣿⣷⣼⣿⣿⣿⣧ ]],
  [[ ⠘⢿⣿⣿⣿⣿⣿⡇⢿⡿⠿⠦⣤⣈⣙⡛⠿⠿⠿⣿⣿⣿⣿⠿⠿⠟⠛⡀ ]],
  [[ ⠄⠄⠉⠻⢿⣿⣿⣷⣬⣙⠳⠶⢶⣤⣍⣙⡛⠓⠒⠶⠶⠶⠶⠖⢒⣛⣛⠁ ]],
  [[ ⠄⠄⠄⠄⠄⠈⠛⠛⠿⠿⣿⣷⣤⣤⣈⣉⣛⣛⣛⡛⠛⠛⠿⠿⠿⠟⢋ ]],
  [[ ⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠉⠉⣻⣿⣿⣿⣿⡿⠿⠛⠃⠄⠙ ]],
}

alpha.setup(dashboard.config)
