function nodeTranslation = translateExpand(~, nodeTranslation, node, IntegerTensorNames)    % Introduced in opset 8
%   Copyright 2024-2025 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};
shape	= node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarShape, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, shape);
shapeName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('shape');
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareExpandArgs(%s, %s);\n', shapeName, NumDimY, getPackageName(nodeTranslation.GraphTranslation), NumDimX, VarShape),...
    sprintf('%s = %s + zeros(%s);\n', VarY, VarX, shapeName),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareExpandArgs";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

