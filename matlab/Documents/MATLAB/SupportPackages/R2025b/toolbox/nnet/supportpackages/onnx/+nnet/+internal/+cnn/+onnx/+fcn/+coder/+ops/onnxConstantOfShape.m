function [Y, numDimsY] = onnxConstantOfShape(value_, ONNXShape_)
% Returns a DLT tensor with the reverse of the ONNXShape. 
%#codegen

% Copyright 2024 The MathWorks, Inc.

    value       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(value_);
    ONNXShape   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXShape_);
    
    DLTShape = fliplr(ONNXShape(:)');
    numDimsY = numel(DLTShape);
    switch numDimsY
        case 0
            % If shape is empty, output is a scalar
            Y = value;
        case 1
            Y = ones(DLTShape, 1) .* value;
        otherwise
            Y = ones(DLTShape) .* value;
    end
end
