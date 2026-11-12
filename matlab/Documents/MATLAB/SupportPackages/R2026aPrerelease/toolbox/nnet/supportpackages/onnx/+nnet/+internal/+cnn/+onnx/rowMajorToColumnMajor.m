function X = rowMajorToColumnMajor(X, onnxDimVec)
% Convert the ONNX tensor X, which we are to interpret as having dimensions
% dimVec in the row-major system, to a column-major array in MATLAB with
% the same dimensions, and maintaining index-equivalence, that is,
% X(i,j,k...) in the row-major system equals X(i,j,k...) in MATLAB.

%   Copyright 2020 The MathWorks, Inc.    

if numel(onnxDimVec) > 1
    onnxDimVec = onnxDimVec(:)';            % Make sure dim is a row vector.
    X = reshape(X, fliplr(onnxDimVec));         % Declare X to have the reversed given shape (because it's row-major).
    X = permute(X, numel(onnxDimVec):-1:1);     % Convert data ordering to col-major, and to the correct shape.
end
end
