function [dlY, numDimsY] = onnxGatherND(dlX_, dlI_, batch_dims_, numDimsX_, numDimsIndices_)
    % Implements the ONNX GatherND operator
    % X and indices passed in reverse ONNX format. 
    % ONNX representation: [<batchShape>,<tupleHolderShape>,<slice>]
    % MATLAB representation: [<slice>,<tupleHolderShape>,<batchShape>]
    %#codegen

    % Copyright 2024 The MathWorks, Inc.
    
    dlX             = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dlX_);
    dlI__           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(dlI_);
    batch_dims      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(batch_dims_);
    numDimsX        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    numDimsIndices  = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsIndices_);

    r = numDimsX;
    q = numDimsIndices;
    b = batch_dims;
    k = size(dlI__,1);   % In ONNX, shape of I is [<batchShape>, <tupleHolderShape>, k], so k is first in reverse-ONNX.
    
    % r>=1 and q>=1 are to be honored. 
    % b < min(q,r) is to be honored
    % The k should have a value between 1(inclusive) and rank "r-b" (inclusive)
    % asserts should execute in matlab version
    
    % Convert the indices stored in indices from origin-0 to origin-1
    dlI__ = dlI__ + 1;
    dlI___ = flip(dlI__,1);
    
    % calculate numDimsY 
    numDimsY = max(1,q+r-k-1-b);

    % Calculate the indice shape in the dlX
    indiceShape = size(dlI___,2:q-b);
    if k==r-b
        % Flatten out tupleHolderShape and batchShape in indices.
        dlI = reshape(dlI___, k, []);    % [k N]
        % Compute linear indices of indexs to gather
        ICell = cell(1,k);
        coder.unroll();
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end

        % Compute the indices of gathered data in each batch
        gatherShape_ = size(dlX, 1:r-b);
        if numel(gatherShape_)<2
            gatherShape = [gatherShape_, ones(1,2-numel(gatherShape_))];
        else
            gatherShape = gatherShape_;
        end
        gatherSize = prod(gatherShape);
        linIdx = sub2ind(gatherShape, ICell{:});

        % Add batch indices to each slice
        [linIdx_, batchShape] = iAddBatchIndex(dlX,r, b, linIdx, gatherSize);

        dlY_ = dlX(linIdx_);                                                              % dlY in shape [indices Size * batch Size]
        dlY = reshape(dlY_,[indiceShape, batchShape]);                                   % dlY in shape [<indices shape>, <batchs shape>]
    else
        % Flatten out tupleHolderShape and batchShape in indices.
        dlI = reshape(dlI___, k, []);                                                      % shape is [k N]
        % Compute linear indices of indexs to gather
        ICell = cell(1,k);
        coder.unroll();
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end

        % Compute the indices of gathered data in each batch
        subSpaceShape_ = size(dlX, r-b-k+1:r-b);                                         % The k dims of dlX to apply the indices.
        if numel(subSpaceShape_)<2
            subSpaceShape = [subSpaceShape_, ones(1,2-numel(subSpaceShape_))];
        else
            subSpaceShape = subSpaceShape_;
        end
        subspaceLinIdx = sub2ind(subSpaceShape, ICell{:});                              % Linear indices as if they indexed only the last k dims.
        leadDataShape = size(dlX, 1:r-b-k);
        numelLeadingData = prod(leadDataShape);                                         % Total number of elements in the first r-k dimensions of dlX.
        linIdxMat = (subspaceLinIdx(:)-1)'*numelLeadingData + (1:numelLeadingData)';    % Adding a row to a column to get a matrix!
        linIdx = linIdxMat(:);
        % Add the batch indices to each indices
        gatherShape = size(dlX, 1:r-b);
        gatherSize = prod(gatherShape);
        [linIdx_, batchShape] = iAddBatchIndex(dlX,r,b,linIdx,gatherSize);

        dlY_ = dlX(linIdx_);                                                              % dlY in shape [r, indices Size * batch Size]
        dlY = reshape(dlY_,[leadDataShape, indiceShape, batchShape]);                    % dlY in shape [r,<indices shape>, <batchs shape>]
    end

    function [linIdx_, batchShape] = iAddBatchIndex(dlX, r, b, linIdx, gatherSize)
        if b == 0
            batchShape = 1;
            linIdx_ = linIdx;
        else
            % Add the batch indices to each indices
            batchShape_ = size(dlX, r-b+1:r);                                            % The last b dims of dlX.
            if numel(batchShape_)<2
                batchShape = [batchShape_, ones(1,2-numel(batchShape_))];
            else
                batchShape = batchShape_;
            end
            batchSize = prod(batchShape);
            batchSpaceLinIdx_ = 1:batchSize;
            batchSpaceLinIdx = repmat(batchSpaceLinIdx_, size(linIdx,1)/batchSize, 1);
            linIdx_ = ((batchSpaceLinIdx(:)-1)'*gatherSize)' + linIdx;                   % Adding a row to a column to get a matrix!
        end
    end
end