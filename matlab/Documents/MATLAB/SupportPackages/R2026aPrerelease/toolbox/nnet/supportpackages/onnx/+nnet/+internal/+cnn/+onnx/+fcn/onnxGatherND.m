function [dlY, numDimsY] = onnxGatherND(dlX, dlI, batch_dims, numDimsX, numDimsIndices)
    % Implements the ONNX GatherND operator
    % X and indices passed in reverse ONNX format. 
    % ONNX representation: [<batchShape>,<tupleHolderShape>,<slice>]
    % MATLAB representation: [<slice>,<tupleHolderShape>,<batchShape>]
    
    % Copyright 2024 The MathWorks, Inc.

    dlX = stripdims(dlX);
    dlI = stripdims(dlI);
    r = numDimsX;
    q = numDimsIndices;
    b = batch_dims;
    k = size(dlI,1);   % In ONNX, shape of I is [<batchShape>, <tupleHolderShape>, k], so k is first in reverse-ONNX.
    
    % r>=1 and q>=1 are to be honored. 
    assert(r>=1 && q>=1,message('nnet_cnn_onnx:onnx:RankSmallerThanOne',"onnxGatherND"))
    % b < min(q,r) is to be honored
    assert(b<min(q,r),message("nnet_cnn_onnx:onnx:BatchDimsLargerThanRank", "onnxGatherND"));
    % The k should have a value between 1(inclusive) and rank "r-b" (inclusive)
    assert(k>= 1 && k<=(r-b),message("nnet_cnn_onnx:onnx:GatherNDInvalidK", "onnxGatherND"));
    
    % Convert the indices stored in indices from origin-0 to origin-1
    dlI = dlI + 1;
    dlI = flip(dlI,1);
    
    % calculate numDimsY 
    numDimsY = max(1,q+r-k-1-b);
    % Calculate the indice shape in the dlX
    indiceShape = size(dlI,2:q-b);
    if k==r-b
        % Flatten out tupleHolderShape and batchShape in indices.
        dlI = reshape(dlI, k, []);    % [k N]
        % Compute linear indices of indexs to gather
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end
        % Compute the indices of gathered data in each batch
        gatherShape = size(dlX, 1:r-b);
        if numel(gatherShape)<2
            gatherShape = [gatherShape, ones(1,2-numel(gatherShape))];
        end
        gatherSize = prod(gatherShape);
        linIdx = sub2ind(gatherShape, ICell{:});
        % Add batch indices to each slice
        [linIdx,batchShape] = addBatchIndice(dlX,r, b,linIdx,gatherSize);

        dlY = dlX(linIdx);                                                              % dlY in shape [indices Size * batch Size]
        dlY = reshape(dlY,[indiceShape, batchShape]);                                   % dlY in shape [<indices shape>, <batchs shape>]
    else
        % Flatten out tupleHolderShape and batchShape in indices.
        dlI = reshape(dlI, k, []);                                                      % shape is [k N]
        % Compute linear indices of indexs to gather
        for i = k:-1:1
            ICell{i} = dlI(i,:)';
        end
        % Compute the indices of gathered data in each batch
        subSpaceShape = size(dlX, r-b-k+1:r-b);                                         % The k dims of dlX to apply the indices.
        if numel(subSpaceShape)<2
            subSpaceShape = [subSpaceShape, ones(1,2-numel(subSpaceShape))];
        end
        subspaceLinIdx = sub2ind(subSpaceShape, ICell{:});                              % Linear indices as if they indexed only the last k dims.
        leadDataShape = size(dlX, 1:r-b-k);
        numelLeadingData = prod(leadDataShape);                                         % Total number of elements in the first r-k dimensions of dlX.
        linIdxMat = (subspaceLinIdx(:)-1)'*numelLeadingData + (1:numelLeadingData)';    % Adding a row to a column to get a matrix!
        linIdx = linIdxMat(:);
        % Add the batch indices to each indices
        gatherShape = size(dlX, 1:r-b);
        gatherSize = prod(gatherShape);
        [linIdx,batchShape] = addBatchIndice(dlX,r,b,linIdx,gatherSize);

        dlY = dlX(linIdx);                                                              % dlY in shape [r, indices Size * batch Size]
        dlY = reshape(dlY,[leadDataShape, indiceShape, batchShape]);                    % dlY in shape [r,<indices shape>, <batchs shape>]
    end

    function [linIdx,batchShape] = addBatchIndice(dlX,r, b,linIdx,gatherSize)
        if b == 0
            batchShape = 1;
            return;
        else
            % Add the batch indices to each indices
            batchShape = size(dlX, r-b+1:r);                                            % The last b dims of dlX.
            if numel(batchShape)<2
                batchShape = [batchShape, ones(1,2-numel(batchShape))];
            end
            batchSize = prod(batchShape);
            batchSpaceLinIdx = 1:batchSize;
            batchSpaceLinIdx = repmat(batchSpaceLinIdx,size(linIdx,1)/batchSize,1);
            linIdx = ((batchSpaceLinIdx(:)-1)'*gatherSize)' + linIdx;                   % Adding a row to a column to get a matrix!
        end
    end
end