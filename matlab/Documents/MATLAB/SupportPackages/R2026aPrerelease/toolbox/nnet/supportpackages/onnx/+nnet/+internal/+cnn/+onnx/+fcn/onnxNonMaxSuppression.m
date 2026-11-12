function [Y, numDimsY] = onnxNonMaxSuppression(boxes, scores, max_output_boxes_per_class, ...
    iou_threshold, score_threshold, center_point_box)
% Implements the ONNX NonMaxSuppression operator

% Returns a 3-by-P array where each column contains [batchIdx; classIdx,
% boxIdx]. This is the reverse of ONNX NonMaxSuppression which returns a
% P-by-3 array where each row contains: [batchIdx, classIdx, boxIdx]. 

% Copyright 2020-2021 The MathWorks, Inc.

if isempty(boxes)
    Y = dlarray([]);
    numDimsY = 2;
    return
end

% If either of the first two inputs is a gpuArray, make sure both of them
% are.
if isgpuarray(boxes) && ~isgpuarray(scores)
    scores = gpuArray(scores);
end
if isgpuarray(scores) && ~isgpuarray(boxes)
    boxes = gpuArray(boxes);
end

% Extract data from input dlarray objects.
boxes = iExtractData(boxes);
scores = iExtractData(scores);
max_output_boxes_per_class = iExtractData(max_output_boxes_per_class);
iou_threshold = iExtractData(iou_threshold);
score_threshold = iExtractData(score_threshold);
center_point_box = iExtractData(center_point_box);
% The ONNX NMS operator's box input is B-by-S-by-4 and the score input is
% B-by-C-by-S. This function receives data in the reverse order.
[numCoords, numBoxes, numBatches]         = size(boxes);
[numScores, numClasses, numBatchesScores] = size(scores);
assert(numCoords == 4);
assert(numBatches == numBatchesScores);
assert(numBoxes == numScores);
% Permute boxes to M-by-4-by-numBatches
boxes = permute(boxes,[2 1 3]);
if center_point_box
    % The center point box format is [xcenter, ycenter, w, h]. Convert to
    % [x1 y1 w h].
    bboxes = boxes;
    bboxes(:,1:2,:) = boxes(:,1:2,:) - bboxes(:,[3 4],:)./2;
else
    % ONNX box format is [y1 x1 y2 x2] in spatial coordinate and define
    % any diagonal pair. The order of the pairs is not defined. Compute the
    % min and max coordinates.
    xmin = min(boxes(:,[2 4],:),[],2);
    xmax = max(boxes(:,[2 4],:),[],2);
    ymin = min(boxes(:,[1 3],:),[],2);
    ymax = max(boxes(:,[1 3],:),[],2);
    % Convert min and max coordinates to [x y w h].
    bboxes = [xmin ymin xmax-xmin ymax-ymin];
end

% Perform NMS across batches.
selectedIndex = cell(numClasses,numBatches,1);
for ii = 1:numBatches
    % Create the full index list and select boxes to process for current
    % batch.
    idx = (1:numBoxes)';

    % Indices are returned as gpuArray if input boxes are on the GPU.
    if isgpuarray(boxes)
        idx = gpuArray(idx);
    end
    
    b = bboxes(:,:,ii);
    s = scores(:,:,ii);
   
    % Process boxes for each class.
    for jj = 1:numClasses
        % Keep boxes above score threshold.
        keep = s(:,jj) > score_threshold;
        boxesAfterScoreThreshold = b(keep,:);
        scoresAfterScoreThreshold = s(keep,jj);
        
        % Track original indices of boxes that were kept.
        idx1 = idx(keep);

        % selectStrongestBbox only supports valid box inputs with width and
        % height > 0. However, ONNX NMS allows these types of boxes and it
        % does not suppress them in its output. Here, we remove invalid
        % boxes, apply selectStrongestBbox, and then add the invalid boxes
        % back to mimic ONNX NMS output.
        %
        % Select valid boxes and keep track of valid box indices.
        valid = all(boxesAfterScoreThreshold(:,[3 4],:) > 0, 2);
        validInd = find(valid);
        validBoxes = boxesAfterScoreThreshold(valid,:);
        validScores = scoresAfterScoreThreshold(valid,:);
        
        if isempty(validBoxes)
            selectedIndex{jj,ii} = zeros(0,0);
            continue
        end
        
        [~,~,index] = selectStrongestBbox(validBoxes,validScores,...
            'RatioType','Union',...
            'OverlapThreshold',iou_threshold,...
            'NumStrongest',max_output_boxes_per_class);
        
        % Append invalid indices to the output of selectStrongestBbox.
        invalidInd = find(~valid);
        index = [validInd(index); invalidInd];

        % Reorder indices by score.
        [~, ord] = sort(scoresAfterScoreThreshold(index),'descend');
        index = index(ord);
        % Get index into original input boxes before score threshold.
        index = idx1(index);
        % Pack indices as M1-by-3 matrices and subtract 1 to return 0-based
        % indices.
        selectedIndex{jj,ii} = [repmat([ii jj],numel(index),1) index] - 1;
    end
end
% Return indices in a 3-by-numSelectedBoxes.
Y = dlarray(vertcat(selectedIndex{:})');
numDimsY = 2;

    function x = iExtractData(x)
        if isa(x,'dlarray')
            x = extractdata(x);
        end
    end
end

