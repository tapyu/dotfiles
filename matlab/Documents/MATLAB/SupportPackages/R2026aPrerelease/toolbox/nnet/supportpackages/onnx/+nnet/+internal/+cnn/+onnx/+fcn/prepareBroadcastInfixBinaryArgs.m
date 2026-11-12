function B = prepareBroadcastInfixBinaryArgs(B, NumDimsA, ONNXAxis)
% For infix binary operators with broadcasting in Opset 6. 
% Broadcasts B to match the shape of A.
% ONNXAxis is the index of the mutually equal dimension in A (in ONNX
% dimension ordering).

%   Copyright 2021-2025 The MathWorks, Inc.    
MLAxis = NumDimsA - ONNXAxis;
% Fill in the old size of B, with ones pre- and post-pended
newSizeB = ones(1,NumDimsA);
newSizeB(MLAxis-numel(size(B))+1:MLAxis) = size(B); 
B = reshape(B, newSizeB);
end