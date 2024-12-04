-- Pull in the wezterm API
local wezterm = require 'wezterm'
local mux = wezterm.mux
local config = {} -- This table will hold the configuration.

-- help provide clearer error messages
if wezterm.config_builder then
  config = wezterm.config_builder()
end

-- define the leader key
config.leader = { key = 'Space', mods = 'CTRL', timeout_milliseconds = 1000 } -- set CTRL+Space as leader key
-- general settings
config.font_size = 17
-- keybinds
config.keys = {
  -- set pane split
  {
    key = 'v',
    mods = 'LEADER',
    action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' },
  },
  {
    key = 'h',
    mods = 'LEADER',
    action = wezterm.action.SplitVertical { domain = 'CurrentPaneDomain' },
  },
  -- unset default split pane
  {
    key = '"',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = '%',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  -- set pane switch
  {
    key = 'j',
    mods = 'ALT',
    action = wezterm.action.ActivatePaneDirection 'Down',
  },
  {
    key = 'h',
    mods = 'ALT',
    action = wezterm.action.ActivatePaneDirection 'Left',
  },
  {
    key = 'l',
    mods = 'ALT',
    action = wezterm.action.ActivatePaneDirection 'Right',
  },
  {
    key = 'k',
    mods = 'ALT',
    action = wezterm.action.ActivatePaneDirection 'Up',
  },
  -- unset default pane switch
  {
    key = 'LeftArrow',
    mods = 'CTRL|SHIFT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'RightArrow',
    mods = 'CTRL|SHIFT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'UpArrow',
    mods = 'CTRL|SHIFT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'DownArrow',
    mods = 'CTRL|SHIFT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  -- set pane resize
  {
    key = 'j',
    mods = 'LEADER|CTRL',
    action = wezterm.action.AdjustPaneSize { 'Down', 5 },
  },
  {
    key = 'h',
    mods = 'LEADER|CTRL',
    action = wezterm.action.AdjustPaneSize { 'Left', 5 },
  },
  {
    key = 'l',
    mods = 'LEADER|CTRL',
    action = wezterm.action.AdjustPaneSize { 'Right', 5 },
  },
  {
    key = 'k',
    mods = 'LEADER|CTRL',
    action = wezterm.action.AdjustPaneSize { 'Up', 5 },
  },
  -- unset default pane resize 
  {
    key = 'LeftArrow',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'RightArrow',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'UpArrow',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
  {
    key = 'DownArrow',
    mods = 'CTRL|SHIFT|ALT',
    action = wezterm.action.DisableDefaultAssignment,
  },
}

-- create functions is catched on wezterm events
local function init_maximized(cmd)
  local tab, pane, window = mux.spawn_window(cmd or {})
  wezterm.log_info 'Maximize the window'
  window:gui_window():maximize()
end

local function test_dpi(window, pane)
  dim = window:get_dimensions()
  for key, value in pairs(dim) do
      wezterm.log_info(key .. " = " .. tostring(value))
  end
end

-- attache functions to events
wezterm.on('gui-startup', init_maximized)
wezterm.on('window-config-reloaded', test_dpi)

-- and finally, return the configuration to wezterm
return config
