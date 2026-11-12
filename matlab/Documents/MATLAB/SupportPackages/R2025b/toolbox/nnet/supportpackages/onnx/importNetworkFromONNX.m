function Network = importNetworkFromONNX(modelfile, varargin)
% 

% Copyright 2022-2024 The MathWorks, Inc.

if isdeployed
    % Function is being called from a compiled app; throw an error        
    error(message("nnet_cnn:supportpackages:CannotDeployImporterOrExporter", mfilename));
end

%% Call the main function
Network = nnet.internal.cnn.onnx.importNetworkFromONNX(modelfile, varargin{:});
end
