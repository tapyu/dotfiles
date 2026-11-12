function [Y, numDimsY] = onnxScatter(X_, ONNXIndices_, updates_, ONNXAxis_, numDimsX_)
% Implements the ONNX Scatter operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    X                 = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    ONNXIndices       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXIndices_);
    updates           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(updates_);
    ONNXAxis          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
    numDimsX          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

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

    function L_ = scatterLinearIndices(X,dim,Indices)
        % Reduce to 0-based.
        L = Indices;
        L = L - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            L = L * prod(size(X, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of X.
        coder.unroll();
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
        L_ = L(:) + 1;
    end
end