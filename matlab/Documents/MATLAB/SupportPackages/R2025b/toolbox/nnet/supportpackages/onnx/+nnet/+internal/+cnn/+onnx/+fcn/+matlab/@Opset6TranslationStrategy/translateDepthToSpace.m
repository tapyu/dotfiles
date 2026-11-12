function nodeTranslation = translateDepthToSpace(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "blocksize"     "INT"       false    []
    });
% Parse the attributes
[blocksize] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

Y        = node.output{1};
X        = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = permute(depthToSpace(permute(%s,[2 1 3 4]),%d,''DataFormat'', [repmat(''S'',[1 %s-2]),''CB'']), [2 1 3 4]);\n',...
    varY, varX, blocksize, numDimsX),...
    sprintf('%s = 4;\n', numDimsY), ...
    ];
nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

