function [T, I, numDimsT, numDimsI] = onnxTopK11(X_, K_, ONNXAxis_, largest_, ~, numDimsX_)
% Implements the ONNX TopK operator. If largest==1, return the K largest,
% otherwise the K smallest. If sorted==1, return them in sorted order.
% Otherwise the returned order is undefined.
%#codegen

% Copyright 2024 The MathWorks, Inc.

    X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    K          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(K_);
    largest    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(largest_);
    ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
    numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

    if ONNXAxis < 0
        ONNXAxis = ONNXAxis + numDimsX;
    end
    
    % To support largest=0, pass -X to maxk to get the indices of the smallest
    % elements.
    if largest==1
        XtoUse = X;
    else
        XtoUse = -X;
    end
    % To support sorted, do nothing. Our maxk always returns the elements in
    % sorted order. In ONNX, when sorted=0, the order is undefined, meaning
    % that sorted order is still acceptable.
    DLTDim = coder.const(numDimsX - ONNXAxis);
    [I, IlinInd] = maxklinind(XtoUse, K, DLTDim);
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
