function nodeTranslation = translateDropout(~, nodeTranslation, node, ~)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "seed"  "INT"   true    []
    });
% Parse the attributes
seed = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
seedStr = "[]";
if ~isempty(seed)
    seedStr = num2str(seed);
end

% Parse the inputs
X = node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
% The second and third inputs are both optional
if numel(node.input) > 1
    ratioName = node.input{2};
else
    ratioName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Ratio']);
    defaultRatio = 0.5;
    setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', ratioName, nnet.internal.cnn.onnx.fcn.RankedArray(defaultRatio, 0));
end
[VarRatio, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ratioName);
if numel(node.input) > 2
    training_mode = node.input{3};
    if ~isempty(training_mode)
        % Throw a translation warning, telling the user that
        % training_mode will be ignored.
        nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message('nnet_cnn_onnx:onnx:TrainingModeAttrIgnored'));
    end
end
Y = node.output{1};
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

switch numel(node.output)
    case 1
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s] = %sonnxDropout12(%s, %s, %s, Training, %s);\n', ...
            VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, seedStr, VarRatio, NumdimX),...
            ];
    case 2
        Mask = node.output{2};
        [VarMask, NumdimMask] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Mask);
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s, %s, %s] = %sonnxDropout12(%s, %s,%s, Training, %s);\n', ...
            VarY, NumdimY, VarMask, NumdimMask, getPackageName(nodeTranslation.GraphTranslation), VarX, seedStr, VarRatio, NumdimX),...
            ];
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxDropout12";
end

