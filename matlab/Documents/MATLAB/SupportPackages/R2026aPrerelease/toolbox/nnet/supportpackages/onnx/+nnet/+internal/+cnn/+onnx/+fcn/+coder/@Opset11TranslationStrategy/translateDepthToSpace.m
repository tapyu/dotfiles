function nodeTranslation = translateDepthToSpace(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "blocksize"     "INT"       false   []
    "mode"          "STRING"    true    "DCR"
    });
% Parse the attributes
[blocksize, mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = onnxDepthToSpace(X, blocksize);
Y        = node.output{1};
X        = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...    
    % sprintf('%s = permute(depthToSpace(permute(%s,[2 1 3 4]),%d,''DataFormat'', [repmat(''S'',[1 %s-2]),''CB''],''Mode'',''%s'' ), [2 1 3 4]);\n',...
    % VarY, VarX, blocksize, NumDimX, mode),...
    % sprintf('%s = 4;\n', NumDimY), ...
    ];
nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

