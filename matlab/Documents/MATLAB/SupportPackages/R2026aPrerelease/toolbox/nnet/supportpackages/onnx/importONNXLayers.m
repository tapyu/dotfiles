function Layers = importONNXLayers(modelfile, varargin)
%  

% Copyright 2018-2024 The MathWorks, Inc.

if isdeployed
    % Function is being called from a compiled app; throw an error        
    error(message("nnet_cnn:supportpackages:CannotDeployImporterOrExporter", mfilename));
end

%% Call the main function
Layers = nnet.internal.cnn.onnx.importONNXLayers(modelfile, varargin{:});
end
