function [Y, numDimsY] = onnxScatterElements(X_, ONNXIndices_, updates_, ONNXAxis_, reduction_, numDimsX_)
% Implements the ONNX ScatterElements operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    X               = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    ONNXIndices     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXIndices_);
    updates         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(updates_);
    ONNXAxis        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
    reduction       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(reduction_);
    numDimsX        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

    if ONNXAxis<0
        ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
    end
    % Convert axis to DLT axis. ONNXAxis is origin 0 and we index dimensions in
    % reverse ONNX ordering
    mlAxis = numDimsX - ONNXAxis;
    % Convert ONNXIndices to DLT. 
    coder.unroll();
    for i = 1:numel(ONNXIndices)
        if ONNXIndices(i)<0
            ONNXIndices(i) = ONNXIndices(i) + size(X,mlAxis); % Make negative ONNXIndices nonnegative.
        end
    end
    mlIndices = ONNXIndices + 1;
    % Find the linear indices of X into which we scatter the updates
    mlLinearIndices = scatterElementsLinearIndices(X, mlAxis, mlIndices);
    Y = X;
    switch reduction
        case "none"
            Y(mlLinearIndices) = updates;
        case "add"
            Y(mlLinearIndices) = Y(mlLinearIndices)+reshape(updates,[],1);
        case "mul"
            Y(mlLinearIndices) = Y(mlLinearIndices).*reshape(updates,[],1);
        case "max"
            Y(mlLinearIndices) = max(Y(mlLinearIndices),reshape(updates,[],1));
        case "min"
            Y(mlLinearIndices) = min(Y(mlLinearIndices),reshape(updates,[],1));
    end
    numDimsY = numDimsX;
    
        function L = scatterElementsLinearIndices(X,dim,Indices)
            % Reduce to 0-based.
            L_ = Indices;
            L_ = L_ - 1;
            if dim > 1
                % Shift to be the component from the relevant dimension.
                L_ = L_*prod(size(X, 1:(dim-1)));
            end
            sz = 1;
            % Go through the dims of X.
            coder.unroll();
            for d = 1:ndims(X)
                if d ~= dim
                    % Add in the component from this dimension.
                    idx = ((1:size(Indices,d))-1)*sz;                   % Note we take the size of Indices here, not X.
                    szvec = [ones(1, d-1),  numel(idx), 1];
                    L_ = L_ + reshape(idx, szvec);
                end
                % Increase the cumulative size.
                sz = sz * size(X,d);
            end
            % Shift back to being 1-based.
            L = L_(:) + 1;
        end
end
