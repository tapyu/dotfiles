function [dlY, numDimsY] = onnxScatterND(dlX_, dlI_, dlU_, reduction_, numDimsX_, ~, ~)
% Function implementing the ONNX ScatterND operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    dlX         = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dlX_);
    dlI__       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dlI_);
    dlU__       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dlU_);
    reduction   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(reduction_);
    numDimsX    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    
    dlY = dlX;
    numDimsY = numDimsX;
    r = numDimsX;
    k = size(dlI__, 1);   % In ONNX, shape of I is [<tupleHolderShape>, k], so k is first in reverse-ONNX.
    
    % Convert the indices stored in dlI from origin-0 to origin-1
    dlI__ = dlI__ + 1;
    if r==1
        % k==1. This is simple 1D assignment no matter the size of dlI and dlU
        dlY(dlI__(:)) = helpReduction(dlY(dlI__(:)),dlU__(:),reduction);
    elseif k==r
        % dlI size is [r, <tupleHolderShape>]. dlU size is
        % [<tupleHolderShape>]. Flip the first dimension of dlI to convert the
        % onnx indices to reverse-onnx indices. Flatten out tupleHolderShape in
        % dlI and dlU.
        dlI___ = flip(dlI__, 1);
        dlI    = reshape(dlI___, k, []);    % [k N]
        dlU    = dlU__(:)';                % [1 N]
        % Compute linear indices and do assignment
        ICell = cell(1,k);
        coder.unroll();
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end
        linIdx = sub2ind(size(dlX), ICell{:});
        dlY(linIdx) = helpReduction(dlY(linIdx),dlU,reduction);
    else
        % dlI size is [k, <tupleHolderShape>]. dlU size is [r-k,
        % <tupleHolderShape>]. Flip the first dimension of dlI to convert the
        % onnx indices to reverse-onnx indices Loop over flattened
        % tupleHolderShape and do assignments. In ONNX, each assignment sets
        % the trailing r-k dimensions of Y. In MATLAB, we set the leading r-k
        % dimensions.
        dlI___ = flip(dlI__,1);
        dlI = reshape(dlI___, k, []);              % shape is [k N]
        uCell = num2cell(size(dlX, 1:r-k));     % value is {d1 ... d_(r-k)}
        dlU = reshape(dlU__, uCell{:}, []);       % shape is [d1 ... d_(r-k) N]
        % Compute linear indices and do assignment
        ICell = cell(1,k);
        coder.unroll();
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end
        trailingSizeX_ = size(dlX, r-k+1:r);                                             % The last k dims of dlX.
        if numel(trailingSizeX_)<2
            trailingSizeX = [trailingSizeX_, ones(1,2-numel(trailingSizeX_))];
        else
            trailingSizeX = trailingSizeX_;
        end
        subspaceLinIdx = sub2ind(trailingSizeX, ICell{:});                              % Linear indices as if they indexed only the last k dims.
        numelLeadingData = prod(size(dlX, 1:r-k));                                      % Total number of elements in the first r-k dimensions of dlX.
        linIdxMat = (subspaceLinIdx(:)-1)'*numelLeadingData + (1:numelLeadingData)';    % Adding a row to a column to get a matrix!
        linIdx = linIdxMat(:);
        dlY(linIdx) = helpReduction(dlY(linIdx),dlU,reduction);
    end

    function reduced = helpReduction(dlY, dlU, reduction)
        switch reduction
            case "none"
                reduced = dlU;
            case "add"
                reduced = dlY + reshape(dlU,[],1);
            case "mul"
                reduced = dlY .* reshape(dlU,[],1);
            case "max"
                reduced = max(dlY, reshape(dlU,[],1));
            case "min"
                reduced = min(dlY, reshape(dlU,[],1));
        end
    end
end