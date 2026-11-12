function [Y, numDimsY] = onnxScatter(X, ONNXIndices, updates, ONNXAxis, numDimsX)

% Copyright 2020 The MathWorks, Inc.


% Implements the ONNX Scatter operator
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
% Convert axis to DLT dim. ONNXAxis is origin 0 and we index dimensions in
% reverse ONNX ordering
mlDim = numDimsX - ONNXAxis;
mlIndices = ONNXIndices + 1;
% Find the linear indices of X into which we scatter the updates
mlLinearIndices = scatterLinearIndices(X, mlDim, mlIndices);
Y = X;
Y(mlLinearIndices) = updates;
numDimsY = numDimsX;

    function L = scatterLinearIndices(X,dim,Indices)
        % Reduce to 0-based.
        L = Indices;
        L = L - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            L = L*prod(size(X, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of X.
        for d = 1:ndims(X)
            if d ~= dim
                % Add in the component from this dimension.
                idx = ((1:size(Indices,d))-1)*sz;                   % Note we take the size of Indices here, not X.
                szvec = [ones(1, d-1),  numel(idx), 1];
                L = L + reshape(idx, szvec);
            end
            % Increase the cumulative size.
            sz = sz * size(X,d);
        end
        % Shift back to being 1-based.
        L = L(:) + 1;
    end
end