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
Network = nnet.internal.cnn.onnx.importONNXNetwork(modelFolder, 'TargetNetwork', 'dlnetwork', ...
    'PackageName', NameValueArgs.PackageName, 'Namespace', NameValueArgs.Namespace, ...
    'InputDataFormats', NameValueArgs.InputDataFormats, 'OutputDataFormats', NameValueArgs.OutputDataFormats,...
    'GenerateCustomLayers', true, 'FoldConstants', 'deep', 'IsInputForwardONNX', true,...
    'CustomLayerContent', 'groupedOperators', 'GenerateCoderFiles', NameValueArgs.GenerateCoderFiles);