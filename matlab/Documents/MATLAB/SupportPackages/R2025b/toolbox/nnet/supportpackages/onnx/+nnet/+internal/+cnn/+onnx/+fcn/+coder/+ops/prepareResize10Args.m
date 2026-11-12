function [DLTScales, dataFormat, Method, numDimsY] = prepareResize10Args(ONNXScales_, mode_, numDimsX_)
% Prepares arguments for implementing the ONNX Resize-10 operator
%#codegen

%   Copyright 2024 The MathWorks, Inc.    

    ONNXScales    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXScales_);
    mode = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(mode_);
    numDimsX = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    
    % ONNXScales are in ONNX dimension ordering
    DLTScales = flip(ONNXScales(:)');
    switch mode
        case "nearest"
            Method = "nearest";
        case "linear"
            Method = "linear";
    end
    dataFormat = [repmat('S', [1 numDimsX-1]), 'B'];
    numDimsY = numDimsX;
end
