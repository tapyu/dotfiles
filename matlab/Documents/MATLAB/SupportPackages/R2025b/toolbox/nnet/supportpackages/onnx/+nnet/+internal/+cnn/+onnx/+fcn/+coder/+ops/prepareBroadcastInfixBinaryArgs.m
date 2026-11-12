function B = prepareBroadcastInfixBinaryArgs(B_, NumDimsA_, ONNXAxis_)
% For infix binary operators with broadcasting in Opset 6. 
% Broadcasts B to match the shape of A.
% ONNXAxis is the index of the mutually equal dimension in A (in ONNX
% dimension ordering).
%   Copyright 2024 The MathWorks, Inc.    
%#codegen

ONNXAxis = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
NumDimsA = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(NumDimsA_);

MLAxis = NumDimsA - ONNXAxis;
% Fill in the old size of B, with ones pre- and post-pended
newSizeB = ones(1, NumDimsA);
newSizeB(MLAxis:MLAxis+numel(size(B_))) = size(B_); 
B = reshape(B_, newSizeB);
end