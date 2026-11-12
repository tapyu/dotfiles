function nodeTranslation = translateHardmax(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

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
axisTempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('axis');
ITempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('I');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareHardmax13Args(%d, %s);\n', ...
    axisTempName, NumdimY, getPackageName(nodeTranslation.GraphTranslation), axis, NumdimX),...
    sprintf('[~,%s] = max(%s, [], %s, ''linear'');\n', ITempName, VarX, axisTempName),...
    sprintf('%s = %s;\n', VarY, VarX),...
    sprintf('%s(:) = 0;\n', VarY),...
    sprintf('%s(%s) = 1;\n', VarY, ITempName),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareHardmax13Args";
end