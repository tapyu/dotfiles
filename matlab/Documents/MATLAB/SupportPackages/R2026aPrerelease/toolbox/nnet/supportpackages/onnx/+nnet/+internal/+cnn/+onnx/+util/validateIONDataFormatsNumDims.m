function tf = validateIONDataFormatsNumDims(onnxSizeCell, isInputTensor, tensorNum, userLabel, internalLabel)

%   Copyright 2021-2025 The MathWorks, Inc.    
tf = 1;
% Check for mismatch with internally-derived numDims.
if ~isempty(onnxSizeCell)
    fileNumDims = numel(onnxSizeCell);
    userNumDims = strlength(userLabel);
    if fileNumDims ~= userNumDims
        tf = 0;
        if isInputTensor
            msgName = 'nnet_cnn_onnx:onnx:InputDataFormatNumDimsMismatch';
        else
            msgName = 'nnet_cnn_onnx:onnx:OutputDataFormatNumDimsMismatch';
        end
        nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace(msgName, userLabel, tensorNum, "{" + join(string(onnxSizeCell), ", ") + "}");
        return;
    end
   
end
% Check for mismatch with internally-derived format.
if internalLabel~="" && internalLabel~=userLabel
    nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:InputLabelMismatch', tensorNum, userLabel, internalLabel);
    tf = 0;
end
end