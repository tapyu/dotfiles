function nodeTranslation = translateEyeLike(~, nodeTranslation, node, IntegerTensorNames)        % Introduced in opset 9
%   Copyright 2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "dtype"         "INT"       true    []
    "k"             "INT"       true    0
    });
% Parse the attributes
[Dtype, K] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

% Output = EyeLike(Input, Dtype, K)
Output  = node.output{1};
Input	= node.input{1};

[VarOutput, NumDimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Output);
[VarInput, NumDimInput]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Input);

if isempty(Dtype)
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxEyeLike(%s, [], %d, %s);\n', ...
        VarOutput, NumDimOutput, getPackageName(nodeTranslation.GraphTranslation), VarInput, K, NumDimInput),...
        ];
else
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxEyeLike(%s, %d, %d, %s);\n', ...
        VarOutput, NumDimOutput, getPackageName(nodeTranslation.GraphTranslation), VarInput, Dtype, K, NumDimInput),...
        ];
end
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "onnxEyeLike";

if ~isempty(Dtype) && ismember(Dtype, [2,3,4,5,6,7,12,13])
    % Dtype is specified and is an integer type
    nodeTranslation.IntegerOutputTensorNames = string(Output);
elseif ismember(Input, IntegerTensorNames)
    % If Dtype is not specified, the data type of the input tensor is used
    nodeTranslation.IntegerOutputTensorNames = string(Output);
end
end