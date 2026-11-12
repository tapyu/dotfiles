function nodeTranslation = translateHardmax(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = hardmax(X)
Y     = node.output{1};
X     = node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[axis, %s] = %sprepareHardmax13Args(%d, %s);\n', ...
    NumdimY, getPackageName(nodeTranslation.GraphTranslation), axis, NumdimX),...
    sprintf('[~,I] = max(%s, [], axis, ''linear'');\n', VarX),...
    sprintf('%s = %s;\n', VarY, VarX),...
    sprintf('%s(:) = 0;\n', VarY),...
    sprintf('%s(I) = 1;\n', VarY),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareHardmax13Args";
end

