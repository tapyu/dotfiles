function [Indices, numDimsY] = prepareCompressArgs(X_, condition_, ONNXAxis_, numDimsX_)
% Prepares arguments for implementing the ONNX Compress operator
%#codegen

% Copyright 2024 The MathWorks, Inc.
    X            = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);    
    condition__    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(condition_); 
    ONNXAxis     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_); 
    numDimsX     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_); 

    if ONNXAxis < 0
        % No axis was specified, so X will be flattened first
        Indices = [condition__; false(numel(X) - numel(condition__), 1)];
        numDimsY = 1;
    else
        DLTAxis = numDimsX - ONNXAxis;
        condition = [condition__; false(size(X, DLTAxis) - numel(condition__), 1)];
        curShape = size(X);
        Indices = cell(1,numel(curShape));
        coder.unroll();
        for i = 1:numDimsX
            if i ~= DLTAxis
                Indices{i} = 1:uint32(curShape(i));
            else
                Indices{i} = condition;
            end        
        end    
        numDimsY = numDimsX;
    end
end
