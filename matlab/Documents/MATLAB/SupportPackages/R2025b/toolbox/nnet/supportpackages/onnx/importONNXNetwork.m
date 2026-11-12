function Network = importONNXNetwork(modelfile, varargin)
% 

% Copyright 2018-2024 The Mathworks, Inc.

if isdeployed
    % Function is being called from a compiled app; throw an error        
    error(message("nnet_cnn:supportpackages:CannotDeployImporterOrExporter", mfilename));
end

%% Call the main function
Network = nnet.internal.cnn.onnx.importONNXNetwork(modelfile, varargin{:});
end
