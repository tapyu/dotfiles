function [Y, numDimsY] = onnxEyeLike(X, Dtype, K, ~)
% Implements the ONNX EyeLike operator.

% Copyright 2025 The MathWorks, Inc.

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
        Y = zeros([nRows nCols]);
    else
        % Create the vector for the diagonal
        d = ones(diagLen, 1);
        % Create the matrix with the diagonal at offset k
        Y = diag(d, K);
        % Pad/truncate to the correct size
        Y(end+1:nRows, end+1:nCols) = 0; % pad if needed
        Y = Y(1:nRows, 1:nCols);         % trim if needed
    end
    
    if ~isempty(Dtype)
        % If Dtype is specified, change the output type
        % Strings (8) and complex are not supported.
        switch Dtype
            case {1, 10, 16}
                Y = cast(single(Y), 'like', X);
            case 2
                Y = cast(uint8(Y), 'like', X);   
            case 3
                Y = cast(int8(Y), 'like', X);
            case 4
                Y = cast(uint16(Y), 'like', X);
            case 5
                Y = cast(int16(Y), 'like', X);
            case 6
                Y = cast(int32(Y), 'like', X);
            case 7
                Y = cast(int64(Y), 'like', X);
            case 9
                Y = cast(logical(Y), 'like', X);
            case {11, 14, 15}
                Y = cast(double(Y), 'like', X);
            case 12
                Y = cast(uint32(Y), 'like', X);
            case 13
                Y = cast(uint64(Y), 'like', X);
            otherwise
                assert(false);
        end  
    else
        % If Dtype is not specified, data type is that of the input tensor
        Y = cast(Y, 'like', X);
    end
    Y = Y'; % Return in reverse ONNX dimensions
    numDimsY = 2;
end