local dap = require('dap')

dap.adapters.julia = {
  type = "executable",
  command = "/usr/bin/julia $HOME/git/DebugAdapter.jl/src/DebugAdapter.jl",
}

dap.configurations.julia = {
	{
		-- The first three options are required by nvim-dap
		type = "julia",
		request = "launch",
		name = "Launch file",
	},
}
