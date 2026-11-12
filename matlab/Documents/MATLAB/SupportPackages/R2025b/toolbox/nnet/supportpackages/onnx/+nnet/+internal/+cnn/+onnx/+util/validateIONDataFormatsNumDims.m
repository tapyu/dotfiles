function validateIONDataFormatsNumDims(onnxSizeCell, isInputTensor, tensorNum, userLabel)

%   Copyright 2021 The MathWorks, Inc.    

if ~isempty(onnxSizeCell)
    fileNumDims = numel(onnxSizeCell);
    userNumDims = strlength(userLabel);
    if fileNumDims ~= userNumDims
        if isInputTensor
            msgName = 'nnet_cnn_onnx:onnx:InputDataFormatNumDimsMismatch';
        else
            msgName = 'nnet_cnn_onnx:onnx:OutputDataFormatNumDimsMismatch';
        end
        error(message(msgName, userLabel, tensorNum, "{" + join(string(onnxSizeCell), ", ") + "}"));
    end
end
end