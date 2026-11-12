function [Yout, numDimsY] = onnxEyeLike(X_, Dtype_, K_, ~)
% Implements the ONNX EyeLike operator.
%#codegen

% Copyright 2025 The MathWorks, Inc.
X = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
Dtype = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(Dtype_);
K = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(K_);


% Input tensor X must be 2-dimensional
sizeX = size(X);    
nRows = sizeX(2);
nCols = sizeX(1);

% Determine length of the diagonal
if K >= 0
    diagLen = min(nRows, nCols - K);
else
    diagLen = min(nRows + K, nCols);
end

if diagLen <= 0
    % Return zeros if diagonal is out of bounds
    Y__ = zeros([nRows nCols]);
else
    % Create the vector for the diagonal
    d = ones(diagLen, 1);
    % Create the matrix with the diagonal at offset k
    Y_ = diag(d, K);    
    sizeY_ = size(Y_);

    % pad or truncate a row if needed
    if nRows > sizeY_(1)
        row = zeros(1, nCols);
        Y___ = [Y_; row]; 
    elseif nRows < sizeY_(1)
        Y___ = Y_(1:nRows,:);
    else
        Y___ = Y_;
    end

    % pad or truncate a column if needed
    if nCols > sizeY_(2)
        col = zeros(nRows, 1);
        Y__ = [Y___ col]; 
    elseif nCols < sizeY_(2)
        Y__ = Y___(:,1:nCols);
    else
        Y__ = Y___;
    end
end

if ~isempty(Dtype)
    % If Dtype is specified, change the output type
    % Strings (8) and complex are not supported.
    switch Dtype
        case {1, 10, 16}
            Y = single(Y__);
        case 2
            Y = uint8(Y__);   
        case 3
            Y = int8(Y__);
        case 4
            Y = uint16(Y__);
        case 5
            Y = int16(Y__);
        case 6
            Y = int32(Y__);
        case 7
            Y = int64(Y__);
        case 9
            Y = logical(Y__);
        case {11, 14, 15}
            Y = double(Y__);
        case 12
            Y = uint32(Y__);
        case 13
            Y = uint64(Y__);
        otherwise
            assert(false);
    end  
else
    % If Dtype is not specified, data type is that of the input tensor
    Y = cast(Y__, 'like', X);
end
Yout = Y'; % Return in reverse ONNX dimensions
numDimsY = 2;
end