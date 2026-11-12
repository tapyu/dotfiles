function [T, I, numDimsT, numDimsI] = onnxTopK(X_, K_, ONNXAxis_, numDimsX_)
% Implements the ONNX TopK operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    K          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(K_);
    ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
    numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

    if ONNXAxis < 0
        ONNXAxis = ONNXAxis + numDimsX;
    end

    DLTDim = coder.const(numDimsX - ONNXAxis);    
    [I, IlinInd] = maxklinind(X, K, DLTDim);
    % I contains indices for each subscript along the axis dim. IlinInd
    % contains the linear indices into X.
    T = X(IlinInd);
    % T, and I (origin-0), are what the ONNX operator returns:
    I = I - 1;
    numDimsT = numDimsX;
    numDimsI = numDimsX;
    
        function [I, IlinInd] = maxklinind(A,k,dim)
            % Gather indices.
            [~,I] = maxk(A,k,dim);
            % Reduce to 0-based.
            IlinInd = I;
            IlinInd = IlinInd - 1;
            if dim > 1
                % Shift to be the component from the relevant dimension.
                IlinInd = IlinInd*prod(size(A, 1:(dim-1)));
            end
            sz = 1;
            % Go through the dims of A.
            coder.unroll();
            for d = 1:ndims(A)
                if d ~= dim
                    % Add in the component from this dimension.
                    idx = ((1:size(A,d))-1)*sz;
                    szvec = [ones(1, d-1),  numel(idx), 1];
                    IlinInd = IlinInd + reshape(idx, szvec);
                end
                % Increase the cumulative size.
                sz = sz * size(A,d);
            end
            % Shift back to being 1-based.
            IlinInd = IlinInd + 1;
        end
end
