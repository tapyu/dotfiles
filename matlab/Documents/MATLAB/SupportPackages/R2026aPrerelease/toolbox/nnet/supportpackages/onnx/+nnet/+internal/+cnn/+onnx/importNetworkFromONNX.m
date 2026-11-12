function Network = importNetworkFromONNX(modelFolder, NameValueArgs)
% Copyright 2022-2024 The MathWorks, Inc.

arguments 
    modelFolder {mustBeTextScalar}
    NameValueArgs.PackageName {mustBeTextScalar} = ''
    NameValueArgs.Namespace {mustBeTextScalar} = ''
    NameValueArgs.InputDataFormats = ''
    NameValueArgs.OutputDataFormats = ''
    NameValueArgs.GenerateCoderFiles (1,1) logical = true
end 

%% Call to importONNXNetwork
try
    Network = nnet.internal.cnn.onnx.importONNXNetwork(modelFolder, 'TargetNetwork', 'dlnetwork', ...
        'PackageName', NameValueArgs.PackageName, 'Namespace', NameValueArgs.Namespace, ...
        'InputDataFormats', NameValueArgs.InputDataFormats, 'OutputDataFormats', NameValueArgs.OutputDataFormats,...
        'GenerateCustomLayers', true, 'FoldConstants', 'deep', 'IsInputForwardONNX', true,...
        'CustomLayerContent', 'groupedOperators', 'GenerateCoderFiles', NameValueArgs.GenerateCoderFiles);
catch ME
    switch ME.identifier
        % Throws error message with "InitializerDataTypeUnsupported" identifier 
        % from calling function to improve readability.
        case 'nnet_cnn_onnx:onnx:InitializerDataTypeUnsupported'
            throwAsCaller(ME)
        otherwise
            % Rethrows previously caught exception to preserve original
            % stack trace.
            rethrow(ME)
    end
end