function nodeTranslation = translateNonMaxSuppression(~, nodeTranslation, node, ~)         % Introduced in opset 10
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "center_point_box"         "INT"       true    0
    });
% Parse the attributes
[center_point_box] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
Y                           = node.output{1};
boxes                       = node.input{1};
scores                      = node.input{2};
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varBoxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, boxes);
[varScores, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, scores);

% Add all initializer inputs beyond 1 to Nonlearnables
for i=2:numel(node.input)
    tensorName = node.input{i};
end

if numel(node.input) > 2 && ~isempty(node.input{3})
    max_output_boxes_per_class = node.input{3};
    [max_output_boxes_per_class, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, max_output_boxes_per_class);
else
    max_output_boxes_per_class = 'Inf';
end
if numel(node.input) > 3 && ~isempty(node.input{4})
    iou_threshold = node.input{4};
    [iou_threshold, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, iou_threshold);
else
    iou_threshold = '0.5';
end
if numel(node.input) > 4 && ~isempty(node.input{5})
    score_threshold = node.input{5};
    [score_threshold, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, score_threshold);

else
    score_threshold = '0';
end

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...
    % sprintf('[%s, %s] = %sonnxNonMaxSuppression(%s, %s, %s, %s, %s, %d);\n',...
    % varY, numDimsY, getPackageName(nodeTranslation.GraphTranslation), varBoxes, varScores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IntegerOutputTensorNames = string(Y);

end