function Network = importNetworkFromONNX(modelfile, varargin)
% 

% Copyright 2022-2025 The MathWorks, Inc.

%% Call the main function
Network = nnet.internal.cnn.onnx.importNetworkFromONNX(modelfile, varargin{:});
end
