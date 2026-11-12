function nodeTranslation = translateSpaceToDepth(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "blocksize"     "INT"       false    []
    });
% Parse the attributes
[blocksize] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = onnxSpaceToDepth(X, blocksize);
Y        = node.output{1};
X        = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = permute(spaceToDepth(permute(%s,[2 1 3 4]),%d, ''DataFormat'', [repmat(''S'',[1 %s-2]),''CB'']),[2 1 3 4]);\n',...
    VarY, VarX, blocksize, NumDimX),...
    sprintf('%s = 4;\n', NumDimY), ...
    ];
nodeTranslation.MCode                   = string(command);
end

