local is_status_ok, _ = pcall(require, "dap")

if not is_status_ok then
  print("nvim-dap plugin not found!")
	return
end

require("user.dap.dap-ui")
require("user.dap.python")
