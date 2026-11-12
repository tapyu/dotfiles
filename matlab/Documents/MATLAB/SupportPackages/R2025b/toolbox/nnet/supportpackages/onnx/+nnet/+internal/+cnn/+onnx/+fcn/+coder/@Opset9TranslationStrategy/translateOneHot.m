function nodeTranslation = translateOneHot(~, nodeTranslation, node, IntegerTensorNames)        % Introduced in opset 9
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% B = OneHot(Indices, Depth, Values)
B       = node.output{1};
Indices	= node.input{1};
Depth  	= node.input{2};
Values 	= node.input{3};
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarIndices, NumDimIndices]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
[VarDepth, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Depth);
[VarValues, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Values);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...
  ];

% command = [
%     sprintf('%% %s:\n', node.op_type),...
%     sprintf('[%s, %s] = %sonnxOneHot(%s, %s, %s, %d, %s);\n', ...
%     VarB, NumDimB, getPackageName(nodeTranslation.GraphTranslation), VarIndices, VarDepth, VarValues, axis, NumDimIndices),...
%     ];
nodeTranslation.MCode = string(command);
if ismember(Values, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end
end

