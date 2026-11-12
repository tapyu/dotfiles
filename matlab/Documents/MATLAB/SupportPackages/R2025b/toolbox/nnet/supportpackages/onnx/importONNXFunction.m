function params = importONNXFunction(modelfile, outputFunctionName)
% 

% Copyright 2020-2024 The Mathworks, Inc.

if isdeployed
    % Function is being called from a compiled app; throw an error        
    error(message("nnet_cnn:supportpackages:CannotDeployImporterOrExporter", mfilename));
end

%% Call the main function
params = nnet.internal.cnn.onnx.importONNXFunction(modelfile, outputFunctionName);
end
