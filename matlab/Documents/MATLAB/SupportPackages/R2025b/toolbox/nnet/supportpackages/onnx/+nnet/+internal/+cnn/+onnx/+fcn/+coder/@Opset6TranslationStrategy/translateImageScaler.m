function nodeTranslation = translateImageScaler(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Bias : list of floats. Bias applied to each channel, same size as C.
% Scale : float (scalar)
Y = node.output{1};
X = node.input{1};
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeFs = arrayfun(@(a) a.f, node.attribute,'UniformOutput',false);
if ~isempty(attributeNames)
    attributeMap = containers.Map(attributeNames, attributeFs);
end
bias = double(0);
scale = double(1);
if ismember('scale', attributeNames)
    scale =  double(attributeMap('scale'));
end
if ismember('bias', attributeNames)
    bias =  double(attributeMap('bias'));
end
if isempty(bias)
    bias = double(0);
end
ScaleName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'scale']);
BiasName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'bias']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', ScaleName, nnet.internal.cnn.onnx.fcn.RankedArray(scale,0));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', BiasName, nnet.internal.cnn.onnx.fcn.RankedArray(bias,2));

[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varScale, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ScaleName);
[varBias, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, BiasName);
Command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s .* %s + %s;\n', varY, varX, varScale, varBias),...
    sprintf('%s = %s;\n', numDimsY, numDimsX),...
    ];
nodeTranslation.MCode = string(Command);
end

